// Playback loop that exp/client.py injects into /player. It does what
// runOnePlayback (templates/player.html) does: one sendPosition() per
// viewpoint, the next one sent after the previous frame is drawn. It does not
// post /metrics/predict or /movement, and records per frame the profile the
// request actually used. Received JPEGs go to the Python side through the
// tigasSaveFrame binding without holding up the next request. In a session
// (client.py --session) it is called once per run, with the run's index and
// the pause after the previous run (delayMs).
(() => {
    // mulberry32: L2A and LoL+ call Math.random, a fixed seed makes runs
    // under the same network repeat the same decisions.
    function seedRandom(seed) {
        let s = seed >>> 0;
        Math.random = () => {
            s = (s + 0x6D2B79F5) >>> 0;
            let t = s;
            t = Math.imul(t ^ (t >>> 15), t | 1);
            t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    function saveFrame(run, i, profile, blob) {
        return new Promise((resolve, reject) => {
            const fr = new FileReader();
            fr.onload = () => {
                const b64 = fr.result.slice(fr.result.indexOf(',') + 1);
                resolve(window.tigasSaveFrame(run, i, profile, b64));
            };
            fr.onerror = () => reject(fr.error);
            fr.readAsDataURL(blob);
        });
    }

    const now = () => performance.timeOrigin + performance.now();  // epoch ms

    window.tigasRun = async ({ events, abr, seed, run = 0, startAtMs, delayMs = 0, quiet }) => {
        const T = window.TIGAS;
        // The ABRs log several lines per frame; over CDP that costs more
        // than in a browser without devtools.
        if (quiet) console.log = console.debug = () => {};
        if (delayMs > 0) await new Promise(r => setTimeout(r, delayMs));
        const wait = startAtMs - now();
        if (wait > 0) await new Promise(r => setTimeout(r, wait));

        seedRandom(seed);
        T.setAbrKind(abr);  // new ABR object, no state from the warm-up
        performance.setResourceTimingBufferSize(1000);
        performance.clearResourceTimings();

        const camera = T.camera();  // intrinsics and size, fixed for the run
        const frames = [];
        const saves = [];
        const start = now();
        for (let i = 0; i < events.length; i++) {
            T.setPose(events[i]);
            const rec = { i, send: now() };
            try {
                const r = await T.sendPosition();
                rec.profile = Number(r.profile);
                rec.bytes = r.blob.size;
                rec.render_ms = r.renderMs;
                saves.push(saveFrame(run, i, rec.profile, r.blob));
            } catch (e) {
                rec.error = String(e);
            }
            rec.drawn = now();
            const rt = performance.getEntriesByType('resource')
                .filter(e => e.name.endsWith('/render')).pop();
            performance.clearResourceTimings();
            if (rt) {
                rec.req_start = performance.timeOrigin + rt.startTime;
                rec.resp_start = performance.timeOrigin + rt.responseStart;
                rec.resp_end = performance.timeOrigin + rt.responseEnd;
                rec.proto = rt.nextHopProtocol;
            }
            const a = T.abr();
            rec.abr_thr = a.lastThroughputBps;
            rec.abr_profile = a.profile;
            frames.push(rec);
        }
        const end = now();
        await Promise.all(saves);
        return { start, end, camera, frames };
    };
})();
