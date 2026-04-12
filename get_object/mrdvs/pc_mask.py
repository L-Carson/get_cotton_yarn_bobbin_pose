"""点云与分割 mask：稀疏 PCD 投影筛选、有序深度网格筛选、深度去噪、重心。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def load_ascii_pcd_xyz(path: str):
    import numpy as np

    data_mode = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ls = line.strip()
            if not ls:
                continue
            if ls.upper().startswith("DATA"):
                parts = ls.split()
                data_mode = parts[-1] if parts else ""
                break
        if not data_mode or data_mode.upper() != "ASCII":
            raise ValueError(f"需要 ASCII PCD: {path}")
        arr = np.loadtxt(f, dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    n = len(arr) // 3
    if n * 3 != len(arr):
        raise ValueError("PCD 数据列数不是 3 的倍数")
    return arr.reshape(-1, 3)


def erode_mask(mask_u8, iterations: int, kernel_size: int):
    import cv2

    if iterations <= 0:
        return mask_u8
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask_u8, kernel, iterations=iterations)


def filter_depth_outliers(
    pts,
    z_mad_k: float,
    z_min_band_mm: float,
    z_max_band_mm: float,
    passes: int,
):
    import numpy as np

    out = np.asarray(pts, dtype=np.float32)
    if out.size == 0:
        return out

    for _ in range(max(1, passes)):
        if len(out) < 4:
            break
        z = out[:, 2]
        med = np.median(z)
        mad = np.median(np.abs(z - med))
        scale = 1.4826
        band = z_mad_k * scale * mad
        band = max(float(z_min_band_mm), band)
        if z_max_band_mm and z_max_band_mm > 0:
            band = min(band, float(z_max_band_mm))
        sel = np.abs(z - med) <= band
        out = out[sel]
    return out


def centroid_xyz(pts) -> tuple[float, float, float] | None:
    import numpy as np

    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return None
    c = pts.mean(axis=0)
    return float(c[0]), float(c[1]), float(c[2])


def save_pcd_ascii(path: str, pts) -> None:
    import numpy as np

    pts = np.asarray(pts, dtype=np.float32)
    n = pts.shape[0]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA ascii\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for row in pts:
            f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")


def project_and_filter(
    pts,
    mask_u8,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    z_min: float,
    flip_v: bool,
    image_size: tuple[int, int],
):
    import numpy as np

    h, w = mask_u8.shape[:2]
    W, H = image_size
    if (w, h) != (W, H):
        raise ValueError(f"mask 尺寸 {(w,h)} 与 RGB 期望 {(W,H)} 不一致")

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (np.abs(z) >= z_min)
    valid &= z > 0

    u = np.zeros_like(z, dtype=np.float64)
    v = np.zeros_like(z, dtype=np.float64)
    u[valid] = fx * x[valid] / z[valid] + cx
    v[valid] = fy * y[valid] / z[valid] + cy
    if flip_v:
        v = (H - 1) - v

    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    inside = valid & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    mhit = np.zeros(len(pts), dtype=bool)
    mhit[inside] = mask_u8[vi[inside], ui[inside]] > 127
    return pts[mhit]


def dense_txt_filter(txt_path: str, mask_u8, grid_w: int, grid_h: int):
    import numpy as np
    import cv2

    data = np.loadtxt(txt_path, dtype=np.float32)
    if data.size != grid_w * grid_h * 3:
        raise ValueError(
            f"txt 点数 {data.size} != {grid_w}*{grid_h}*3={grid_w*grid_h*3}"
        )
    pts = data.reshape(grid_h, grid_w, 3)
    mh, mw = mask_u8.shape[:2]
    if (mw, mh) != (grid_w, grid_h):
        mask_r = cv2.resize(mask_u8, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)
    else:
        mask_r = mask_u8
    sel = mask_r > 127
    return pts[sel]


def filter_organized_xyz_with_mask(
    xyz_hw,
    mask_u8,
    *,
    erode_iters: int,
    erode_kernel: int,
    no_depth_filter: bool,
    z_mad_k: float,
    z_min_band_mm: float,
    z_max_band_mm: float,
    z_filter_passes: int,
    z_min: float = 1e-3,
):
    """
    从 SDK getPointCloud 得到的有序网格 (H_d, W_d, 3) 与 RGB 同尺寸的 mask 筛选点。
    mask 会自动缩放到与 xyz 一致（最近邻）。
    """
    import cv2
    import numpy as np

    xyz_hw = np.asarray(xyz_hw, dtype=np.float32)
    if xyz_hw.ndim != 3 or xyz_hw.shape[2] != 3:
        raise ValueError("xyz_hw 须为 (H, W, 3)")
    Hd, Wd = xyz_hw.shape[:2]
    Hm, Wm = mask_u8.shape[:2]
    if (Hm, Wm) != (Hd, Wd):
        mask_r = cv2.resize(mask_u8, (Wd, Hd), interpolation=cv2.INTER_NEAREST)
    else:
        mask_r = mask_u8
    mask_r = erode_mask(mask_r, erode_iters, erode_kernel)
    z = xyz_hw[:, :, 2]
    sel = (mask_r > 127) & np.isfinite(z) & (np.abs(z) >= z_min) & (z > 0)
    pts = xyz_hw[sel].reshape(-1, 3)
    if no_depth_filter or len(pts) == 0:
        return pts
    return filter_depth_outliers(
        pts, z_mad_k, z_min_band_mm, z_max_band_mm, z_filter_passes
    )
