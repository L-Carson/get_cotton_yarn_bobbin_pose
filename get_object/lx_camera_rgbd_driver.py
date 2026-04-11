#!/usr/bin/env python3
"""
LxCamera RGB + 对齐点云批量采集

用法（在项目任意工作目录下执行均可，会保存到当前工作目录）:
  python3 /path/to/lx_camera_rgbd_driver.py 100

默认按 IP 打开相机（192.168.100.82），枚举不到设备时仍会尝试直连。
其他 IP: --param 192.168.x.x ；按索引打开: --open-mode index --param 0

表示连续采集并保存 100 组：rgb_000001.png … rgb_000100.png 与
pointcloud_000001.pcd … pointcloud_000100.pcd。

依赖: opencv-python；LxCameraSDK 为厂商 wheel 内容，可放在:
  - 本脚本同目录的 LxCameraSDK/（与 get_object 里结构相同）
  - 或仓库 get_object/ 下（脚本会自动把该目录加入 sys.path）
动态库 libLxCameraApi.so / .dll 放在上述 LxCameraSDK 目录内，或通过环境变量
LX_CAMERA_SDK 指定绝对路径。

对齐: DEPTH_TO_RGB + SYNC_FRAME（与 SDK C 示例一致）。

点云投影到 RGB 像素: 须使用 **2D 内参**（get2DIntricParam / LX_PTR_2D_INTRIC_PARAM），
对应保存的 rgb_*.png 分辨率。不要用 3D 内参（深度图坐标系/分辨率不同）。
脚本会在输出目录写入 camera_2d_intrinsics.json，供 mask_filter_pointcloud.py 使用。
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _ensure_lxsdk_on_path() -> str | None:
    """将包含 LxCameraSDK 包的目录加入 sys.path，返回该根目录或 None。"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        here,
        os.path.normpath(os.path.join(here, "..", "get_object")),
        os.getcwd(),
    ]
    for root in candidates:
        init_py = os.path.join(root, "LxCameraSDK", "__init__.py")
        if os.path.isfile(init_py):
            if root not in sys.path:
                sys.path.insert(0, root)
            return root
    return None


_PKG_ROOT = _ensure_lxsdk_on_path()

try:
    import cv2
except ImportError as e:
    print("请先安装: pip install opencv-python", file=sys.stderr)
    raise SystemExit(1) from e

try:
    from LxCameraSDK import LxCamera
    from LxCameraSDK.lx_camera_define import (
        LX_CAMERA_FEATURE,
        LX_OPEN_MODE,
        LX_RGBD_ALIGN_MODE,
        LX_STATE,
    )
except ImportError as e:
    print(
        "无法导入 LxCameraSDK。请将 wheel 中的 LxCameraSDK 文件夹放在本脚本同目录，"
        "或放在仓库 get_object/ 下。",
        file=sys.stderr,
    )
    raise SystemExit(1) from e


def _find_native_lib(explicit: str) -> str | None:
    if explicit.strip() and os.path.isfile(explicit):
        return os.path.abspath(explicit)
    env = os.environ.get("LX_CAMERA_SDK", "").strip()
    if env and os.path.isfile(env):
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    search_dirs: list[str] = []
    if _PKG_ROOT:
        search_dirs.append(os.path.join(_PKG_ROOT, "LxCameraSDK"))
    search_dirs.extend(
        [
            here,
            os.path.join(here, "LxCameraSDK"),
            os.path.normpath(os.path.join(here, "..", "get_object", "LxCameraSDK")),
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


def _check(camera: LxCamera, state: LX_STATE, what: str) -> None:
    if state != LX_STATE.LX_SUCCESS:
        msg = camera.DcGetErrorString(state)
        raise RuntimeError(f"{what} 失败: {msg}")


def _save_rgb_intrinsics_json(camera: LxCamera, handle, out_dir: str) -> str | None:
    """
    读取 **2D（RGB）内参**，用于把 DEPTH_TO_RGB 对齐后的相机坐标点投影到 RGB 像素 (u,v)。

    说明: SDK 另有 get3DIntricParam（深度/点云网格分辨率），那是深度图上的针孔模型；
    分割掩码画在 rgb_*.png 上，必须用 2D 内参。
    """
    st, K, dist = camera.get2DIntricParam(handle)
    if st != LX_STATE.LX_SUCCESS or not K:
        print(
            f"警告: 读取 2D(RGB) 内参失败: {camera.DcGetErrorString(st)}",
            flush=True,
        )
        return None

    payload: dict = {
        "_comment": (
            "fx,fy,cx,cy 为 RGB(2D) 相机内参，用于将点云 (X,Y,Z) 投影到 rgb_*.png 的像素坐标；"
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

    st_w, wv = camera.DcGetIntValue(
        handle, LX_CAMERA_FEATURE.LX_INT_2D_IMAGE_WIDTH
    )
    st_h, hv = camera.DcGetIntValue(
        handle, LX_CAMERA_FEATURE.LX_INT_2D_IMAGE_HEIGHT
    )
    if st_w == LX_STATE.LX_SUCCESS and st_h == LX_STATE.LX_SUCCESS:
        payload["rgb_width"] = int(wv.cur_value)
        payload["rgb_height"] = int(hv.cur_value)

    path = os.path.join(out_dir, "camera_2d_intrinsics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(
        f"已保存 RGB(2D) 内参 -> {path}（用于点云投影到 PNG 像素；非 3D 深度内参）",
        flush=True,
    )
    return path


def _save_pair(
    camera,
    handle,
    out_dir: str,
    index_1based: int,
    rgb_image,
) -> None:
    """在当前工作目录下写入 rgb_XXXXXX.png 与 pointcloud_XXXXXX.pcd。"""
    stem = f"{index_1based:06d}"
    rgb_path = os.path.join(out_dir, f"rgb_{stem}.png")
    pcd_path = os.path.join(out_dir, f"pointcloud_{stem}.pcd")
    if rgb_image.ndim == 3 and rgb_image.shape[2] >= 3:
        to_save = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    else:
        to_save = rgb_image
    if not cv2.imwrite(rgb_path, to_save):
        raise RuntimeError(f"写入 RGB 失败: {rgb_path}")
    st = camera.DcSaveXYZ(handle, pcd_path)
    if st != LX_STATE.LX_SUCCESS:
        raise RuntimeError(
            f"保存点云失败: {camera.DcGetErrorString(st)} -> {pcd_path}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="批量保存对齐的 RGB 与点云（保存到当前工作目录）"
    )
    parser.add_argument(
        "count",
        type=int,
        help="保存组数（每组 1 张 PNG + 1 个 PCD）",
    )
    parser.add_argument(
        "--sdk",
        default="",
        help="可选：libLxCameraApi.so / .dll 绝对路径；默认自动在 LxCameraSDK 目录查找",
    )
    parser.add_argument(
        "--open-mode",
        choices=("index", "ip", "sn", "id"),
        default="ip",
        help="打开方式；按 IP 打开时即使枚举不到设备也会尝试直连（默认 ip）",
    )
    parser.add_argument(
        "--param",
        default="192.168.100.82",
        help="打开参数：索引 / IP / SN / ID（默认本机相机 IP）",
    )
    parser.add_argument(
        "--out",
        default="",
        help="输出目录，默认当前工作目录",
    )
    args = parser.parse_args()
    if args.count < 1:
        print("count 须为正整数", file=sys.stderr)
        return 1

    lib = _find_native_lib(args.sdk)
    if not lib:
        print(
            "未找到 libLxCameraApi.so（或 Windows 下 LxCamera 的 .dll）。"
            "请将其放在 LxCameraSDK 目录内，或设置环境变量 LX_CAMERA_SDK。",
            file=sys.stderr,
        )
        return 1

    out_dir = os.path.abspath(args.out or os.getcwd())
    os.makedirs(out_dir, exist_ok=True)

    camera = LxCamera(lib)
    print("API 版本:", camera.DcGetApiVersion())
    print("动态库:", lib)
    print("保存目录:", out_dir)
    camera.DcSetInfoOutput(2, False, "./", 0)

    state, _dev_list, dev_num = camera.DcGetDeviceList()
    _check(camera, state, "DcGetDeviceList")
    print(f"搜索到设备数量: {dev_num}")
    # 按索引打开必须能枚举到设备；按 IP/SN/ID 可在枚举为空时仍尝试直连（GigE 发现失败但路由可达时常有）
    if dev_num <= 0 and args.open_mode == "index":
        print("未枚举到任何相机，无法按索引打开。请检查网线/权限或改用 --open-mode ip。", file=sys.stderr)
        return 1
    if dev_num <= 0:
        print(
            "提示: 枚举未找到设备，仍将按 "
            f"{args.open_mode}={args.param!r} 尝试打开…",
            flush=True,
        )

    mode_map = {
        "index": LX_OPEN_MODE.OPEN_BY_INDEX,
        "ip": LX_OPEN_MODE.OPEN_BY_IP,
        "sn": LX_OPEN_MODE.OPEN_BY_SN,
        "id": LX_OPEN_MODE.OPEN_BY_ID,
    }
    state, handle, device_info = camera.DcOpenDevice(mode_map[args.open_mode], args.param)
    _check(camera, state, "DcOpenDevice")
    print(
        f"已打开设备: id={device_info.id} ip={device_info.ip} sn={device_info.sn}"
    )

    try:
        _check(
            camera,
            camera.DcSetBoolValue(
                handle, LX_CAMERA_FEATURE.LX_BOOL_ENABLE_2D_STREAM, True
            ),
            "开启 2D (RGB) 流",
        )
        _check(
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
                "警告: RGBD 对齐模式设置未成功（部分机型可能不支持）:",
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

        _check(camera, camera.DcStartStream(handle), "DcStartStream")

        _save_rgb_intrinsics_json(camera, handle, out_dir)

        saved = 0
        goal = args.count
        while saved < goal:
            state, frame_ptr = camera.getFrame(handle)
            if state != LX_STATE.LX_SUCCESS or frame_ptr is None:
                if state == LX_STATE.LX_E_RECONNECTING:
                    continue
                continue

            st, rgb_image = camera.getRGBImage(frame_ptr)
            if st != LX_STATE.LX_SUCCESS or rgb_image is None:
                continue

            st, _ = camera.getPointCloud(handle)
            if st != LX_STATE.LX_SUCCESS:
                continue

            saved += 1
            _save_pair(camera, handle, out_dir, saved, rgb_image)
            print(f"已保存 {saved}/{goal}  -> rgb_{saved:06d}.png / pointcloud_{saved:06d}.pcd", flush=True)

        st = camera.DcStopStream(handle)
        if st != LX_STATE.LX_SUCCESS:
            print("DcStopStream:", camera.DcGetErrorString(st), flush=True)
    finally:
        try:
            camera.DcCloseDevice(handle)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
