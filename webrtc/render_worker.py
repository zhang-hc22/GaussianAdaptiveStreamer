#!/usr/bin/env python
"""Render worker of the WebRTC transport. tigas_sender (sparkrtc branch
tigas, examples/tigas) sends it one viewpoint at a time over a Unix socket;
it renders the viewpoint and replies with the frame in I420.

Protocol (examples/tigas/render_client.h):
- request: one JSON line {"i": <viewpoint>, "body": <the /render body>};
- reply: a 20-byte little-endian header (int32 i, uint32 width,
  uint32 height, float64 render_ms), then width * height * 3 / 2 bytes of
  I420 (Y, U, V, no padding). width 0 means the request failed and no pixels
  follow.

The render is render_image_raw's (render.py) at profile 0: the same
rasterization call and the same uint8 RGB. RGB goes to I420 on the GPU with
the coefficients of webrtc/yuv.py, and only the I420 is copied to the CPU.
render_ms spans rasterization start to the I420 on the CPU, the span
X-Render-Time-Ms covers in /render (raster, downsample, transfer), with the
conversion in place of the downsample.

One sender at a time. When it disconnects the worker waits for the next
one, so one worker (one model load) serves a whole experiment, like the
QUIC server.

--fake-yuv serves frame (i mod N) of a raw I420 file instead of rendering,
for tests on a machine without the model or a GPU.
"""
import argparse
import json
import mmap
import os
import socket
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from yuv import M, OFFSET, i420_size  # noqa: E402

HEADER = struct.Struct("<iIId")


class Renderer:
    """Renders a /render body at profile 0 to I420 bytes on the GPU."""

    def __init__(self, scene):
        sys.path.insert(0, REPO)
        import torch
        from gsplat import rasterization
        from models import get_model
        from render import create_viewmat
        self.torch, self.rasterization = torch, rasterization
        self.get_model, self.create_viewmat = get_model, create_viewmat
        self.m = torch.tensor(M, dtype=torch.float32, device="cuda")
        self.offset = torch.tensor(OFFSET, dtype=torch.float32, device="cuda")
        get_model(scene).load()

    def __call__(self, body):
        torch = self.torch
        w, h = int(body.get("width", 800)), int(body.get("height", 600))
        fx, fy = float(body.get("fx", 1300.0)), float(body.get("fy", 800.0))
        cx, cy = float(body.get("cx", 400.0)), float(body.get("cy", 300.0))
        model = self.get_model(body.get("modelId"))
        device, means, quats, scales, opacities, shs = model.acquire()
        viewmat = self.create_viewmat(
            float(body.get("angle", 180)), float(body.get("elevation", 0)), float(body.get("x", 0)),
            float(body.get("y", 0)), float(body.get("z", 5.0))).to(device).unsqueeze(0)
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32,
                         device=device).unsqueeze(0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                colors, _, _ = self.rasterization(
                    means=means, quats=quats, scales=scales, opacities=opacities, colors=shs,
                    viewmats=viewmat, Ks=K, width=w, height=h, packed=False, sh_degree=0,
                    backgrounds=None, render_mode="RGB")
        finally:
            model.release()
        rgb = (colors[0].clamp(0, 1) * 255).byte()  # as render_image_raw
        torch.cuda.synchronize()
        t_raster = time.perf_counter()
        with torch.no_grad():
            x = rgb.float()
            y = x @ self.m[0] + self.offset[0]
            blk = x.reshape(h // 2, 2, w // 2, 2, 3).mean(dim=(1, 3))
            u = blk @ self.m[1] + self.offset[1]
            v = blk @ self.m[2] + self.offset[2]
            planes = [p.round().clamp(0, 255).to(torch.uint8).reshape(-1) for p in (y, u, v)]
            i420 = torch.cat(planes)
        torch.cuda.synchronize()
        t_convert = time.perf_counter()
        data = i420.cpu().numpy().tobytes()
        t_transfer = time.perf_counter()
        parts = {"raster_ms": (t_raster - t0) * 1e3, "convert_ms": (t_convert - t_raster) * 1e3,
                 "transfer_ms": (t_transfer - t_convert) * 1e3}
        return w, h, data, (t_transfer - t0) * 1e3, parts


class FakeRenderer:
    """Frame (i mod N) of a raw I420 file, after sleeping --fake-render-ms."""

    def __init__(self, path, width, height, render_ms):
        self.w, self.h, self.render_ms = width, height, render_ms
        self.size = i420_size(width, height)
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.n = len(self.mm) // self.size

    def __call__(self, body, i):
        t0 = time.perf_counter()
        k = i % self.n
        data = self.mm[k * self.size:(k + 1) * self.size]
        left = self.render_ms / 1e3 - (time.perf_counter() - t0)
        if left > 0:
            time.sleep(left)
        return self.w, self.h, data, (time.perf_counter() - t0) * 1e3, {"fake_frame": k}


def serve(sock, render, fake, log):
    while True:
        conn, _ = sock.accept()
        print("render_worker: sender connected", flush=True)
        n = 0
        try:
            for line in conn.makefile("rb"):
                t_recv = time.time() * 1e3
                req = json.loads(line)
                i = int(req["i"])
                try:
                    if fake:
                        w, h, data, render_ms, parts = render(req["body"], i)
                    else:
                        w, h, data, render_ms, parts = render(req["body"])
                except Exception as e:  # reply, so the sender is not left waiting
                    print(f"render_worker: viewpoint {i}: {type(e).__name__}: {e}", flush=True)
                    conn.sendall(HEADER.pack(i, 0, 0, -1.0))
                    continue
                conn.sendall(HEADER.pack(i, w, h, render_ms) + data)
                n += 1
                if log:
                    log.write(json.dumps({"i": i, "t_recv": round(t_recv, 3),
                                          "t_sent": round(time.time() * 1e3, 3),
                                          "render_ms": round(render_ms, 3),
                                          **{k: round(v, 3) if isinstance(v, float) else v
                                             for k, v in parts.items()}}) + "\n")
                    log.flush()
        except (BrokenPipeError, ConnectionResetError) as e:
            print(f"render_worker: sender gone ({type(e).__name__})", flush=True)
        finally:
            conn.close()
        print(f"render_worker: sender disconnected after {n} frames", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--socket", default="/tmp/tigas_render.sock")
    ap.add_argument("--scene", default="train", help="model loaded at start")
    ap.add_argument("--log", help="ndjson, one line per request")
    ap.add_argument("--fake-yuv", help="raw I420 file to serve instead of rendering")
    ap.add_argument("--fake-size", default="1280x720", help="WxH of --fake-yuv")
    ap.add_argument("--fake-render-ms", type=float, default=12.0)
    a = ap.parse_args()

    if a.fake_yuv:
        w, h = map(int, a.fake_size.split("x"))
        render = FakeRenderer(a.fake_yuv, w, h, a.fake_render_ms)
        print(f"render_worker: fake, {render.n} frames of {a.fake_size} from {a.fake_yuv}", flush=True)
    else:
        render = Renderer(a.scene)
        # the first rasterization compiles kernels; keep it out of the runs
        render({"modelId": a.scene, "width": 1280, "height": 720, "cx": 640, "cy": 360})
        print(f"render_worker: model {a.scene} loaded", flush=True)

    if os.path.exists(a.socket):
        os.unlink(a.socket)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(a.socket)
    sock.listen(1)
    print(f"render_worker: listening on {a.socket}", flush=True)
    log = open(a.log, "a") if a.log else None
    try:
        serve(sock, render, bool(a.fake_yuv), log)
    finally:
        sock.close()
        os.unlink(a.socket)


if __name__ == "__main__":
    main()
