// Playback loop that exp/client.py --mode wt injects into /player-wt. The
// page streams on its own: every 10 ms it sends the current viewpoint with
// the profile of its ThroughputABR, and the server renders the latest
// viewpoint 60 times a second and sends each frame as datagrams
// (gs_app.py). Like the page's runOnePlayback, this loop only sets the
// viewpoints, one every eventMs; the page waits sleep(40) after each one,
// which drifts, here viewpoint k is set at start + k * eventMs.
//
// It records every pose message (sends) and every frame of the playback
// (frames, one per image id, also those that never completed), through the
// hooks at the end of the page. The first frame drawn for each viewpoint
// goes to the Python side through the tigasSaveFrame binding. After the last
// viewpoint's eventMs it keeps listening tailMs for frames still on the way.
(() => {
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
    const sleep = ms => new Promise(r => setTimeout(r, ms));

    window.tigasRunWT = async ({ events, eventMs = 40, tailMs = 1000, run = 0, startAtMs, quiet }) => {
        const T = window.TIGAS;
        if (quiet) console.log = console.debug = () => {};
        const wait = startAtMs - now();
        if (wait > 0) await sleep(wait);

        T.abr().reset();   // as the page's experiment start does
        const camera = T.camera();
        let cur = -1;          // viewpoint currently set
        const sends = [];      // [timestemp, viewpoint, profile] of every pose message
        const sentAt = new Map();   // timestemp -> [viewpoint, profile]
        const frames = new Map();   // image id -> record
        const shown = new Array(events.length).fill(null);   // image id first drawn per viewpoint
        const saves = [];

        T.onSend = m => {
            if (cur < 0) return;
            sends.push([m.timestemp, cur, Number(m.profile)]);
            sentAt.set(m.timestemp, [cur, Number(m.profile)]);
        };
        T.onChunk = (h, t, blob) => {
            let f = frames.get(h.id);
            if (!f) {
                const s = sentAt.get(h.ts);
                if (!s) return;   // rendered from a pose sent before the playback
                f = { id: h.id, ts: h.ts, i: s[0], profile: s[1], total: h.total, chunks: 0, bytes: 0,
                      first: t, render_ms: h.renderMs };
                frames.set(h.id, f);
            }
            f.chunks++;
            f.bytes += h.bytes;
            f.last = t;
            if (blob) {   // the page has all chunks and draws the frame now
                f.drawn = t;
                if (shown[f.i] === null) {
                    shown[f.i] = f.id;
                    saves.push(saveFrame(run, f.i, f.profile, blob));
                }
            }
        };

        const start = now();
        const applied = [];
        for (let k = 0; k < events.length; k++) {
            const w = start + k * eventMs - now();
            if (w > 0) await sleep(w);
            T.setPose(events[k]);
            cur = k;
            applied.push(now());
        }
        const w = start + events.length * eventMs - now();
        if (w > 0) await sleep(w);
        const end = now();
        await sleep(tailMs);
        T.onSend = T.onChunk = null;
        await Promise.all(saves);
        return { start, end, camera, applied, sends, frames: [...frames.values()], shown };
    };
})();
