// Page script of exp/client.py --mode webrtc, injected into a blank page.
// It signs in to peerconnection_server, answers the offer of tigas_sender
// (sparkrtc branch tigas, examples/tigas), sends viewpoints over a negotiated
// DataChannel and reads every decoded frame with MediaStreamTrackProcessor.
//
// Messages to the sender: {i, body}, body being the /render JSON of /player
// (templates/player.html) at profile 0; warm-up viewpoints have i < 0.
// At the end {type: "bye"}. For each frame it renders, the sender replies
// {i, rtp_ts, render_ms, t_push}; rtp_ts equals the frame's
// metadata().rtpTimestamp, which is how a decoded frame finds its viewpoint.
// Either one can arrive first.
//
// Pacing: "tms" sends viewpoint i at startAtMs + tMs_i; "closed" sends the
// next viewpoint when the frame of the previous one has been read (as /player
// does with requests), or after closedTimeoutMs.
// Warm-up, from the DataChannel opening to startAtMs, always timed:
// "reverse" sends viewpoints J, J-1, ..., 1 at startAtMs - tMs_j, so the
// motion runs into viewpoint 0; "pose0" sends viewpoint 0 at the mean
// viewpoint interval.
//
// Every saveEvery-th viewpoint's first frame goes to Python as I420 through
// the tigasSaveYuv binding. Such a viewpoint that gets no frame (lost frames,
// a closed-pacing timeout, the end of the run) gets a stand-in instead: the
// frame read before the first frame of a later viewpoint, i.e. the one on
// screen while its frame was due. Frames decode in RTP order, so once a later
// viewpoint's frame is read, its own will not come. All times are epoch ms.
(() => {
    const now = () => performance.timeOrigin + performance.now();
    const sleepUntil = t => new Promise(r => setTimeout(r, Math.max(0, t - now())));
    const epoch = t => (t === undefined || t === null ? null : performance.timeOrigin + t);

    class Signaling {
        constructor(base, name) {
            this.base = base;
            this.name = name;
            this.id = -1;
            this.closed = false;
            this.sendChain = Promise.resolve();
            this.onmessage = null;
        }

        async signIn() {
            const r = await fetch(`${this.base}/sign_in?${encodeURIComponent(this.name)}`);
            if (!r.ok) throw new Error(`sign_in: HTTP ${r.status}`);
            this.id = Number(r.headers.get('Pragma'));
            await r.text();
            this.waitLoop();
        }

        // Hanging GET: a message from another peer (Pragma: its id) or a
        // change in the peer list (Pragma: our id).
        async waitLoop() {
            while (!this.closed) {
                let r, text;
                try {
                    r = await fetch(`${this.base}/wait?peer_id=${this.id}`);
                    text = await r.text();
                } catch (e) {
                    if (this.closed) return;
                    await new Promise(res => setTimeout(res, 100));
                    continue;
                }
                const from = Number(r.headers.get('Pragma'));
                if (r.ok && from !== this.id && this.onmessage) this.onmessage(from, text);
            }
        }

        // One message at a time, in order.
        send(to, text) {
            this.sendChain = this.sendChain
                .then(() => fetch(`${this.base}/message?peer_id=${this.id}&to=${to}`,
                                  { method: 'POST', body: text }))
                .then(r => { if (!r.ok) throw new Error(`message: HTTP ${r.status}`); })
                .catch(e => console.error(`signaling: ${e}`));
            return this.sendChain;
        }

        async signOut() {
            this.closed = true;
            await this.sendChain;
            try { await fetch(`${this.base}/sign_out?peer_id=${this.id}`); } catch (e) { /* gone */ }
        }
    }

    window.tigasRunWebRTC = async (cfg) => {
        const {
            signaling, modelId, camera, startAtMs, pacing = 'tms', warmup = 'reverse',
            closedTimeoutMs = 2000, tailMs = 1000, saveEvery = 12, statsMs = 100,
            display = true, connectTimeoutMs = 8000, quiet = true,
        } = cfg;
        // client.py hands the events over beforehand (window.tigasEvents):
        // copying 2167 of them into this call took ~100 ms after t0.
        const events = cfg.events || window.tigasEvents;
        if (quiet) console.log = console.debug = () => {};
        const N = events.length;
        const times = {};          // connection milestones
        const poses = [];          // one record per viewpoint of the run
        const warm = [];           // warm-up messages
        const metas = [];          // the sender's per-frame messages
        const decoded = [];        // every frame read from the track
        const shown = [];          // requestVideoFrameCallback records
        const rstats = [];         // getStats every statsMs
        const saves = [];
        const standins = [];       // {i, from_rtp}: sampled viewpoints without a frame
        let pendingAtRead = 0;     // frames read before their message
        let prev = null;           // the frame read last: {frame (a clone), rtp}
        let lastI = -1;            // the highest viewpoint whose frame was read
        const metaByRtp = new Map();
        const pendingByRtp = new Map();  // frames read before their message
        const waiters = new Map();       // closed pacing: i -> resolve
        let stopped = false;

        const bodyOf = ev => ({
            modelId, angle: ev.angle, elevation: ev.elevation, x: ev.x, y: ev.y, z: ev.z,
            fx: camera.fx, fy: camera.fy, cx: camera.cx, cy: camera.cy,
            width: camera.width, height: camera.height, profile: 0,
        });

        // --- connection -----------------------------------------------------
        const sig = new Signaling(signaling, 'tigas_page');
        const pc = new RTCPeerConnection({ iceServers: [], bundlePolicy: 'max-bundle', rtcpMuxPolicy: 'require' });
        const dc = pc.createDataChannel('tigas', { negotiated: true, id: 0, ordered: true });
        let senderId = -1;
        const pendingCandidates = [];
        let msgChain = Promise.resolve();

        async function onSignal(from, text) {
            if (text === 'BYE') return;
            const msg = JSON.parse(text);
            if (msg.type === 'offer') {
                senderId = from;
                times.offer = now();
                await pc.setRemoteDescription({ type: 'offer', sdp: msg.sdp });
                for (const c of pendingCandidates.splice(0)) await pc.addIceCandidate(c);
                await pc.setLocalDescription(await pc.createAnswer());
                sig.send(from, JSON.stringify({ type: 'answer', sdp: pc.localDescription.sdp }));
                times.answer = now();
            } else if (msg.candidate) {
                const c = { candidate: msg.candidate, sdpMid: msg.sdpMid, sdpMLineIndex: msg.sdpMLineIndex };
                if (pc.remoteDescription) await pc.addIceCandidate(c); else pendingCandidates.push(c);
            }
        }
        sig.onmessage = (from, text) => {
            msgChain = msgChain.then(() => onSignal(from, text))
                .catch(e => console.error(`signal handling: ${e}`));
        };
        pc.onicecandidate = e => {
            if (e.candidate && e.candidate.candidate && senderId >= 0) {
                sig.send(senderId, JSON.stringify({
                    sdpMid: e.candidate.sdpMid, sdpMLineIndex: e.candidate.sdpMLineIndex,
                    candidate: e.candidate.candidate,
                }));
            }
        };
        pc.oniceconnectionstatechange = () => { times[`ice_${pc.iceConnectionState}`] ??= now(); };
        pc.onconnectionstatechange = () => { times[`pc_${pc.connectionState}`] ??= now(); };

        // --- frames ---------------------------------------------------------
        function saveYuv(i, w, h, data, kind = 'frame') {
            return new Promise((resolve, reject) => {
                const fr = new FileReader();
                fr.onload = () => {
                    const b64 = fr.result.slice(fr.result.indexOf(',') + 1);
                    resolve(window.tigasSaveYuv(i, w, h, b64, kind));
                };
                fr.onerror = () => reject(fr.error);
                fr.readAsDataURL(new Blob([data]));
            });
        }

        // Sampled viewpoints after lastI and before k that have no frame.
        function gap(k) {
            const out = [];
            for (let s = Math.floor(lastI / saveEvery + 1) * saveEvery; s < Math.min(k, poses.length);
                 s += saveEvery) {
                if (poses[s] && poses[s].dec === undefined) out.push(s);
            }
            return out;
        }

        // Stand-ins for the gap before viewpoint k; before = {rtp, w, h, data}
        // of the frame read just before k's.
        function fillGap(k, before) {
            const ss = gap(k);
            if (ss.length && before && before.data) {
                for (const s of ss) {
                    standins.push({ i: s, from_rtp: before.rtp });
                    saves.push(saveYuv(s, before.w, before.h, before.data, 'standin'));
                }
            }
            lastI = Math.max(lastI, k);
        }

        async function prevCopy() {
            return prev ? { rtp: prev.rtp, w: prev.frame.codedWidth, h: prev.frame.codedHeight,
                            data: await copyI420(prev.frame) } : null;
        }

        // The first frame of viewpoint i (or of a warm-up message).
        function onFrameFor(meta, rec, data) {
            rec.i = meta.i;
            if (meta.i < 0) {
                const w = warm[-meta.i - 1];
                if (w && w.dec === undefined) Object.assign(w, { dec: rec.dec, rtp_ts: rec.rtp });
                return;
            }
            const p = poses[meta.i];
            if (!p || p.dec !== undefined) return;
            Object.assign(p, {
                dec: rec.dec, rtp_ts: rec.rtp, recv: rec.recv, render_ms: meta.render_ms,
                t_push: meta.t_push, t_meta: meta.t_meta, frame: rec.n, width: rec.w, height: rec.h,
            });
            if (meta.i % saveEvery === 0 && data) saves.push(saveYuv(meta.i, rec.w, rec.h, data));
            const wake = waiters.get(meta.i);
            if (wake) { waiters.delete(meta.i); wake(true); }
        }

        dc.onmessage = e => {
            const m = JSON.parse(e.data);
            m.t_meta = now();
            metas.push(m);
            metaByRtp.set(m.rtp_ts, m);
            const pend = pendingByRtp.get(m.rtp_ts);
            if (pend) {
                pendingByRtp.delete(m.rtp_ts);
                if (m.i >= 0) fillGap(m.i, pend.before);
                onFrameFor(m, pend.rec, pend.data);
            }
        };

        async function copyI420(frame) {
            if (frame.format !== 'I420' && frame.format !== 'I420A') return null;
            const w = frame.codedWidth, h = frame.codedHeight;
            const cw = (w + 1) >> 1, ch = (h + 1) >> 1;
            const buf = new Uint8Array(w * h + 2 * cw * ch);
            // tightly packed Y, U, V (no alpha plane for I420A)
            await frame.copyTo(buf, { layout: [
                { offset: 0, stride: w }, { offset: w * h, stride: cw },
                { offset: w * h + cw * ch, stride: cw }] });
            return buf;
        }

        async function readFrames(track) {
            const reader = new MediaStreamTrackProcessor({ track, maxBufferSize: 30 }).readable.getReader();
            while (true) {
                const { value: frame, done } = await reader.read();
                if (done) return;
                const t = now();
                const md = frame.metadata ? frame.metadata() : {};
                const rec = {
                    n: decoded.length, rtp: md.rtpTimestamp, dec: t, w: frame.codedWidth,
                    h: frame.codedHeight, fmt: frame.format, recv: epoch(md.receiveTime),
                    cap: epoch(md.captureTime),
                };
                decoded.push(rec);
                times.first_frame ??= t;
                try {
                    const meta = metaByRtp.get(rec.rtp);
                    if (meta) {
                        if (meta.i >= 0 && gap(meta.i).length) fillGap(meta.i, await prevCopy());
                        else if (meta.i >= 0) lastI = Math.max(lastI, meta.i);
                        const save = meta.i >= 0 && meta.i % saveEvery === 0 && poses[meta.i]
                            && poses[meta.i].dec === undefined;
                        onFrameFor(meta, rec, save ? await copyI420(frame) : null);
                    } else {
                        // Its viewpoint is not known yet; keep the previous frame
                        // too in case it closes a gap.
                        pendingAtRead++;
                        if (pendingByRtp.size >= 8) pendingByRtp.delete(pendingByRtp.keys().next().value);
                        pendingByRtp.set(rec.rtp, {
                            rec, data: await copyI420(frame),
                            before: gap(Infinity).length ? await prevCopy() : null,
                        });
                    }
                } finally {
                    if (prev) prev.frame.close();
                    prev = { frame: frame.clone(), rtp: rec.rtp };
                    frame.close();
                }
            }
        }

        function watchDisplay(track) {
            const v = document.createElement('video');
            v.muted = true;
            v.autoplay = true;
            v.playsInline = true;
            v.srcObject = new MediaStream([track]);
            document.body.appendChild(v);
            v.play().catch(e => console.error(`video.play: ${e}`));
            const cb = (_t, md) => {
                shown.push({
                    rtp: md.rtpTimestamp, presented: epoch(md.presentationTime),
                    expected: epoch(md.expectedDisplayTime), recv: epoch(md.receiveTime),
                    n: md.presentedFrames,
                });
                if (!stopped) v.requestVideoFrameCallback(cb);
            };
            v.requestVideoFrameCallback(cb);
        }

        let readerDone = null;
        pc.ontrack = e => {
            times.track = now();
            readerDone = readFrames(e.track).catch(err => console.error(`reader: ${err}`));
            if (display) watchDisplay(e.track);
        };

        async function pollStats() {
            while (!stopped) {
                const t = now();
                const rep = await pc.getStats();
                let pairId = null, pair = null;
                rep.forEach(s => { if (s.type === 'transport' && s.selectedCandidatePairId) pairId = s.selectedCandidatePairId; });
                if (pairId) pair = rep.get(pairId);
                const keep = [];
                rep.forEach(s => {
                    if ((s.type === 'inbound-rtp' && s.kind === 'video') || s.type === 'transport'
                        || s.type === 'data-channel' || s.type === 'codec' || s.id === pairId
                        || (pair && (s.id === pair.localCandidateId || s.id === pair.remoteCandidateId))) {
                        keep.push(s);
                    }
                });
                rstats.push({ t, stats: keep });
                await sleepUntil(t + statsMs);
            }
        }

        // --- connect --------------------------------------------------------
        const dcOpen = new Promise((resolve, reject) => {
            dc.onopen = () => { times.dc_open = now(); resolve(); };
            setTimeout(() => reject(new Error(`no data channel within ${connectTimeoutMs} ms`)), connectTimeoutMs);
        });
        times.start = now();
        await sig.signIn();
        times.signed_in = now();
        await dcOpen;
        const statsDone = pollStats().catch(e => console.error(`stats: ${e}`));

        // --- warm-up --------------------------------------------------------
        function sendWarm(j) {
            const i = -1 - warm.length;
            warm.push({ i, j, send: now() });
            dc.send(JSON.stringify({ i, body: bodyOf(events[j]) }));
        }
        if (warmup === 'reverse') {
            let J = 0;
            const t = now();
            while (J + 1 < N && startAtMs - events[J + 1].tMs > t) J++;
            for (let j = J; j >= 1; j--) {
                await sleepUntil(startAtMs - events[j].tMs);
                sendWarm(j);
            }
        } else if (warmup === 'pose0') {
            const dt = events[N - 1].tMs / (N - 1);
            for (let t = now(); t < startAtMs - dt / 2; t += dt) {
                await sleepUntil(t);
                sendWarm(0);
            }
        }

        // --- playback -------------------------------------------------------
        await sleepUntil(startAtMs);
        const start = now();
        function sendPose(i) {
            poses[i] = { i, send: now() };
            dc.send(JSON.stringify({ i, body: bodyOf(events[i]) }));
        }
        const waitFrame = (i, ms) => new Promise(resolve => {
            if (poses[i].dec !== undefined) return resolve(true);
            waiters.set(i, resolve);
            setTimeout(() => { if (waiters.delete(i)) resolve(false); }, ms);
        });
        if (pacing === 'tms') {
            for (let i = 0; i < N; i++) {
                await sleepUntil(startAtMs + events[i].tMs);
                sendPose(i);
            }
            const deadline = now() + tailMs;
            while (now() < deadline && poses.some(p => p.dec === undefined)) {
                await new Promise(r => setTimeout(r, 20));
            }
        } else {
            for (let i = 0; i < N; i++) {
                sendPose(i);
                if (!(await waitFrame(i, closedTimeoutMs))) poses[i].error = 'timeout';
            }
        }
        const end = now();

        // --- close ----------------------------------------------------------
        for (const p of poses) if (p.dec === undefined && !p.error) p.error = 'no frame';
        stopped = true;
        await statsDone;
        dc.send(JSON.stringify({ type: 'bye' }));
        await new Promise(r => setTimeout(r, 200));
        pc.close();
        await sig.signOut();
        await Promise.race([readerDone, new Promise(r => setTimeout(r, 500))]);
        // sampled viewpoints after the last frame read
        if (gap(Infinity).length) fillGap(Infinity, await prevCopy());
        if (prev) prev.frame.close();
        await Promise.all(saves);
        return { start, end, camera, times, poses, warm, metas, decoded, shown, rstats, standins,
                 pendingAtRead };
    };
})();
