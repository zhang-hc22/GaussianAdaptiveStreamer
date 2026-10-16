"""RGB <-> I420 with BT.601 limited-range coefficients, shared by the render
worker (forward, on the GPU) and exp/metrics.py (forward for the
no-encoding ceiling, inverse for the decoded frames).

Forward, with R, G, B in [0, 255]:

    [Y, Cb, Cr] = M @ [R, G, B] + [16, 128, 128]

Cb and Cr are computed from the mean RGB of each 2x2 block. Values are
rounded to the nearest integer and clipped to [0, 255]. The inverse uses
inv(M) and repeats each chroma sample over its 2x2 block.
"""
import numpy as np

M = np.array([[65.481, 128.553, 24.966],
              [-37.797, -74.203, 112.0],
              [112.0, -93.786, -18.214]]) / 255.0
OFFSET = np.array([16.0, 128.0, 128.0])
M_INV = np.linalg.inv(M)


def i420_size(width, height):
    return width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)


def rgb_to_i420(rgb):
    """HxWx3 uint8 RGB (H, W even) -> I420 bytes (Y, U, V planes)."""
    h, w, _ = rgb.shape
    x = rgb.astype(np.float64)
    y = x @ M[0] + OFFSET[0]
    blk = x.reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3))
    u = blk @ M[1] + OFFSET[1]
    v = blk @ M[2] + OFFSET[2]
    planes = [np.clip(np.rint(p), 0, 255).astype(np.uint8) for p in (y, u, v)]
    return b"".join(p.tobytes() for p in planes)


def i420_to_rgb(buf, width, height):
    """I420 bytes -> HxWx3 uint8 RGB."""
    w, h = width, height
    a = np.frombuffer(buf, dtype=np.uint8, count=i420_size(w, h))
    y = a[:w * h].reshape(h, w).astype(np.float64)
    cw, ch = w // 2, h // 2
    u = a[w * h:w * h + cw * ch].reshape(ch, cw).astype(np.float64)
    v = a[w * h + cw * ch:].reshape(ch, cw).astype(np.float64)
    u = u.repeat(2, axis=0).repeat(2, axis=1)
    v = v.repeat(2, axis=0).repeat(2, axis=1)
    yuv = np.stack([y, u, v], axis=-1) - OFFSET
    rgb = yuv @ M_INV.T
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
