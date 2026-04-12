#!/usr/bin/env python3
"""
LxCamera RGB + 对齐点云批量采集

用法（在项目任意工作目录下执行均可，会保存到当前工作目录）:
  python3 /path/to/lx_camera_rgbd_driver.py 100

默认按 IP 打开相机（192.168.100.82），枚举不到设备时仍会尝试直连。
其他 IP: --param 192.168.x.x ；按索引打开: --open-mode index --param 0

表示连续采集并保存 100 组：rgb_000001.png … rgb_000100.png 与
pointcloud_000001.pcd … pointcloud_000100.pcd。

核心逻辑见 mrdvs.rgbd；本脚本为命令行入口，便于单独调试。
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import cv2
except ImportError as e:
    print("请先安装: pip install opencv-python", file=sys.stderr)
    raise SystemExit(1) from e

from mrdvs import rgbd as R

R.ensure_lxsdk_on_path()

try:
    from LxCameraSDK import LxCamera
    from LxCameraSDK.lx_camera_define import LX_OPEN_MODE, LX_STATE
except ImportError as e:
    print(
        "无法导入 LxCameraSDK。请将 wheel 中的 LxCameraSDK 文件夹放在本脚本同目录，"
        "或放在仓库 get_object/ 下。",
        file=sys.stderr,
    )
    raise SystemExit(1) from e


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

    lib = R.find_native_lib(args.sdk)
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
    R.check(camera, state, "DcGetDeviceList")
    print(f"搜索到设备数量: {dev_num}")
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
    R.check(camera, state, "DcOpenDevice")
    print(
        f"已打开设备: id={device_info.id} ip={device_info.ip} sn={device_info.sn}"
    )

    try:
        R.configure_rgbd_alignment(camera, handle)
        R.check(camera, camera.DcStartStream(handle), "DcStartStream")

        ipath = R.save_rgb_intrinsics_json(camera, handle, out_dir)
        if ipath:
            print(
                f"已保存 RGB(2D) 内参 -> {ipath}（用于点云投影到 PNG 像素；非 3D 深度内参）",
                flush=True,
            )

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
            R.save_pair_png_pcd(camera, handle, out_dir, saved, rgb_image)
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
