"""LxCamera：打开设备、对齐流、抓取 RGB + 有序点云、保存 2D 内参。"""

from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

_PKG_ROOT: str | None = None


def ensure_lxsdk_on_path() -> str | None:
    """将包含 LxCameraSDK 的目录加入 sys.path。"""
    global _PKG_ROOT
    here = os.path.dirname(os.path.abspath(__file__))
    get_object_dir = os.path.normpath(os.path.join(here, ".."))
    candidates = [get_object_dir, here, os.getcwd()]
    for root in candidates:
        init_py = os.path.join(root, "LxCameraSDK", "__init__.py")
        if os.path.isfile(init_py):
            if root not in sys.path:
                sys.path.insert(0, root)
            _PKG_ROOT = root  # type: ignore[misc]
            return root
    return None


def find_native_lib(explicit: str = "") -> str | None:
    if explicit.strip() and os.path.isfile(explicit):
        return os.path.abspath(explicit)
    env = os.environ.get("LX_CAMERA_SDK", "").strip()
    if env and os.path.isfile(env):
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    get_object_dir = os.path.normpath(os.path.join(here, ".."))
    search_dirs: list[str] = []
    if _PKG_ROOT:
        search_dirs.append(os.path.join(_PKG_ROOT, "LxCameraSDK"))
    search_dirs.extend(
        [
            get_object_dir,
            os.path.join(get_object_dir, "LxCameraSDK"),
            here,
            os.path.join(here, "LxCameraSDK"),
            os.path.normpath(os.path.join(here, "..", "MRDVS", "SDK", "lib", "linux_x64")),
            os.getcwd(),
            os.path.join(os.getcwd(), "LxCameraSDK"),
        ]
    )
    seen: set[str] = set()
    for d in search_dirs:
        d = os.path.normpath(d)
        if d in seen or not os.path.isdir(d):
            continue
        seen.add(d)
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for fn in names:
            if sys.platform.startswith("win"):
                if fn.endswith(".dll") and "LxCamera" in fn:
                    return os.path.join(d, fn)
            else:
                if fn.endswith(".so") and "LxCamera" in fn:
                    return os.path.join(d, fn)
    return None


def check(camera: Any, state: Any, what: str) -> None:
    from LxCameraSDK.lx_camera_define import LX_STATE

    if state != LX_STATE.LX_SUCCESS:
        msg = camera.DcGetErrorString(state)
        raise RuntimeError(f"{what} 失败: {msg}")


def save_rgb_intrinsics_json(camera: Any, handle: Any, out_dir: str) -> str | None:
    from LxCameraSDK.lx_camera_define import LX_CAMERA_FEATURE, LX_STATE

    st, K, dist = camera.get2DIntricParam(handle)
    if st != LX_STATE.LX_SUCCESS or not K:
        return None

    payload: dict = {
        "_comment": (
            "fx,fy,cx,cy 为 RGB(2D) 相机内参，用于将点云投影到 rgb 像素；"
            "与 mask_filter_pointcloud.py --intrinsics-json 配套"
        ),
        "fx": float(K[0]),
        "fy": float(K[1]),
        "cx": float(K[2]),
        "cy": float(K[3]),
    }
    if dist and len(dist) >= 5:
        payload["distortion"] = {
            "d1": float(dist[0]),
            "d2": float(dist[1]),
            "d3": float(dist[2]),
            "d4": float(dist[3]),
            "d5": float(dist[4]),
        }

    st_w, wv = camera.DcGetIntValue(handle, LX_CAMERA_FEATURE.LX_INT_2D_IMAGE_WIDTH)
    st_h, hv = camera.DcGetIntValue(handle, LX_CAMERA_FEATURE.LX_INT_2D_IMAGE_HEIGHT)
    if st_w == LX_STATE.LX_SUCCESS and st_h == LX_STATE.LX_SUCCESS:
        payload["rgb_width"] = int(wv.cur_value)
        payload["rgb_height"] = int(hv.cur_value)

    path = os.path.join(out_dir, "camera_2d_intrinsics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


def get_intrinsics_2d_dict(camera: Any, handle: Any) -> dict | None:
    """内存中获取 fx,fy,cx,cy，不写文件。"""
    from LxCameraSDK.lx_camera_define import LX_STATE

    st, K, _dist = camera.get2DIntricParam(handle)
    if st != LX_STATE.LX_SUCCESS or not K:
        return None
    return {"fx": float(K[0]), "fy": float(K[1]), "cx": float(K[2]), "cy": float(K[3])}


def grab_rgb_xyz(camera: Any, handle: Any):
    """
    取一帧对齐数据：RGB (H,W,3) 与有序点云 (H,W,3)。
    失败时返回 (None, None, state)。
    """
    from LxCameraSDK.lx_camera_define import LX_STATE

    state, frame_ptr = camera.getFrame(handle)
    if state != LX_STATE.LX_SUCCESS or frame_ptr is None:
        return None, None, state

    st, rgb_image = camera.getRGBImage(frame_ptr)
    if st != LX_STATE.LX_SUCCESS or rgb_image is None:
        return None, None, st

    st, points = camera.getPointCloud(handle)
    if st != LX_STATE.LX_SUCCESS or points is None:
        return rgb_image, None, st

    return rgb_image, points, LX_STATE.LX_SUCCESS


def configure_rgbd_alignment(camera: Any, handle: Any) -> None:
    """开启 2D/3D 流、DEPTH_TO_RGB、同步帧；失败时打印警告。"""
    from LxCameraSDK.lx_camera_define import (
        LX_CAMERA_FEATURE,
        LX_RGBD_ALIGN_MODE,
        LX_STATE,
    )

    check(
        camera,
        camera.DcSetBoolValue(handle, LX_CAMERA_FEATURE.LX_BOOL_ENABLE_2D_STREAM, True),
        "开启 2D (RGB) 流",
    )
    check(
        camera,
        camera.DcSetBoolValue(
            handle, LX_CAMERA_FEATURE.LX_BOOL_ENABLE_3D_DEPTH_STREAM, True
        ),
        "开启 3D 深度流",
    )
    try:
        depth_to_rgb = LX_RGBD_ALIGN_MODE.DEPTH_TO_RGB
    except AttributeError:
        depth_to_rgb = 1
    st_align = camera.DcSetIntValue(
        handle,
        LX_CAMERA_FEATURE.LX_INT_RGBD_ALIGN_MODE,
        depth_to_rgb,
    )
    if st_align != LX_STATE.LX_SUCCESS:
        print(
            "警告: RGBD 对齐模式设置未成功:",
            camera.DcGetErrorString(st_align),
            flush=True,
        )
    st_sync = camera.DcSetBoolValue(
        handle, LX_CAMERA_FEATURE.LX_BOOL_ENABLE_SYNC_FRAME, True
    )
    if st_sync != LX_STATE.LX_SUCCESS:
        print(
            "警告: 帧同步未开启:",
            camera.DcGetErrorString(st_sync),
            flush=True,
        )


def save_pair_png_pcd(
    camera: Any,
    handle: Any,
    out_dir: str,
    index_1based: int,
    rgb_image,
) -> None:
    import cv2

    stem = f"{index_1based:06d}"
    rgb_path = os.path.join(out_dir, f"rgb_{stem}.png")
    pcd_path = os.path.join(out_dir, f"pointcloud_{stem}.pcd")
    if rgb_image.ndim == 3 and rgb_image.shape[2] >= 3:
        to_save = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    else:
        to_save = rgb_image
    if not cv2.imwrite(rgb_path, to_save):
        raise RuntimeError(f"写入 RGB 失败: {rgb_path}")
    from LxCameraSDK.lx_camera_define import LX_STATE

    st = camera.DcSaveXYZ(handle, pcd_path)
    if st != LX_STATE.LX_SUCCESS:
        raise RuntimeError(
            f"保存点云失败: {camera.DcGetErrorString(st)} -> {pcd_path}"
        )


# 模块加载时尝试加入 SDK 路径
ensure_lxsdk_on_path()
