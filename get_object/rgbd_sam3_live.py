#!/usr/bin/env python3
"""
实时：相机 RGB + 有序点云 → SAM3 分割 → mask 筛选点云 → 显示 2D 叠加与 3D（可选）及重心。

SAM3 环境（GPU）: Python≥3.12, PyTorch≥2.7, CUDA 12.6+ 对应的 PyTorch（见 pytorch.org）。
其他依赖: 同 lx_camera_rgbd_driver / sam3_exemplar_segment / mask_filter_pointcloud；
可选 3D 窗口: pip install open3d

用法（在 get_object 目录或设置 PYTHONPATH）:
  python3 rgbd_sam3_live.py --text \"cotton yarn bobbin\"
  # GPU: 需驱动与 PyTorch 的 CUDA 版本匹配；否则自动退回 CPU（很慢）。
  # CPU: 未指定 --imgsz 时默认用 640 以减轻延迟；Open3D 默认关闭（--o3d 开启）。

单独调试仍请使用:
  lx_camera_rgbd_driver.py / sam3_exemplar_segment.py / mask_filter_pointcloud.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# 保证可导入 mrdvs（脚本在 get_object 下运行）
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from mrdvs import rgbd as R
from mrdvs import pc_mask as PC
from mrdvs.sam3_seg import (
    Sam3Segmenter,
    explain_cuda_unavailable_if_needed,
    parse_box,
    resolve_device,
    resolve_imgsz_for_runtime,
)


def _merge_masks(masks: list, instance: int):
    """instance==-1 合并全部；否则取指定下标。"""
    import numpy as np

    if not masks:
        return None
    if instance < 0:
        m = np.zeros_like(masks[0], dtype=np.uint8)
        for t in masks:
            m = np.maximum(m, t)
        return m
    if instance >= len(masks):
        return masks[-1]
    return masks[instance]


def _try_open3d_vis():
    try:
        import open3d as o3d

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="filtered XYZ", width=640, height=480)
        geom = o3d.geometry.PointCloud()
        vis.add_geometry(geom)
        return vis, geom
    except Exception:
        return None, None


def main() -> int:
    import cv2
    import numpy as np

    p = argparse.ArgumentParser(description="RGB-D + SAM3 + 点云筛选 实时演示")
    p.add_argument("--sdk", default="", help="libLxCameraApi.so 绝对路径")
    p.add_argument(
        "--open-mode",
        choices=("index", "ip", "sn", "id"),
        default="ip",
    )
    p.add_argument("--param", default="192.168.100.82", help="打开参数")
    p.add_argument("--weights", "-w", default="", help="sam3.pt，默认 get_object/sam3.pt")
    p.add_argument(
        "--text",
        default="cotton yarn bobbin",
        help="SAM3 英文文本提示",
    )
    p.add_argument(
        "--target-bbox",
        default="",
        help="可选 x1 y1 x2 y2，若填则不用 --text",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument(
        "--device",
        default="",
        help="SAM3 设备：0/1/cpu；默认自动选 GPU（有 CUDA 时）",
    )
    p.add_argument("--cpu", action="store_true", help="强制 CPU 推理（忽略 GPU）")
    p.add_argument(
        "--fp32",
        action="store_true",
        help="GPU 上使用 FP32（默认用 FP16 加速）",
    )
    p.add_argument(
        "--instance",
        type=int,
        default=0,
        help="使用第几个实例 mask；-1 表示合并全部",
    )
    p.add_argument("--erode-iters", type=int, default=3)
    p.add_argument("--erode-kernel", type=int, default=5)
    p.add_argument("--no-depth-filter", action="store_true")
    p.add_argument("--z-mad-k", type=float, default=3.5)
    p.add_argument("--z-min-band-mm", type=float, default=25.0)
    p.add_argument("--z-max-band-mm", type=float, default=0.0)
    p.add_argument("--z-filter-passes", type=int, default=2)
    p.add_argument(
        "--o3d",
        action="store_true",
        help="开启 Open3D 点云窗口（默认关，避免 Qt 线程告警并省 CPU）",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="每 N 帧做一次 SAM3（减轻算力）",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=0,
        help="SAM 正方形边长；0=在 CPU 上自动用 640（可用 --no-auto-imgsz 关闭）",
    )
    p.add_argument(
        "--no-auto-imgsz",
        action="store_true",
        help="CPU 上仍使用 Ultralytics 默认 imgsz（更准、更慢）",
    )
    p.add_argument(
        "--auto-cpu-imgsz",
        type=int,
        default=640,
        help="CPU 且未指定 --imgsz 时使用的边长（默认 640，可试 512 更快）",
    )
    p.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="CPU 推理时 torch 线程数，0=自动 min(8, CPU核数)",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile 加速（PyTorch 2+，首次编译较慢）",
    )
    p.add_argument(
        "--no-warmup",
        action="store_true",
        help="跳过启动时一次空图预热（GPU 首帧可能更卡）",
    )
    args = p.parse_args()

    import torch

    explain_cuda_unavailable_if_needed()

    R.ensure_lxsdk_on_path()
    try:
        from LxCameraSDK import LxCamera
        from LxCameraSDK.lx_camera_define import LX_OPEN_MODE, LX_STATE
    except ImportError as e:
        print("无法导入 LxCameraSDK:", e, file=sys.stderr)
        return 1

    lib = R.find_native_lib(args.sdk)
    if not lib:
        print("未找到 libLxCameraApi.so", file=sys.stderr)
        return 1

    wpath = args.weights.strip() or os.path.join(_HERE, "sam3.pt")
    if args.cpu:
        sam_dev = "cpu"
        sam_half: bool | None = False
    else:
        sam_dev = (args.device or "").strip() or resolve_device("")
        sam_half = False if args.fp32 else None

    imgsz_kw, imgsz_note = resolve_imgsz_for_runtime(
        args.imgsz,
        sam_dev,
        auto_cpu_imgsz=max(256, args.auto_cpu_imgsz),
        no_auto=args.no_auto_imgsz,
    )

    if sam_dev == "cpu":
        nt = args.cpu_threads
        if nt <= 0:
            nt = min(8, max(1, (os.cpu_count() or 4)))
        torch.set_num_threads(nt)
        print(f"CPU 推理: torch.set_num_threads({nt})", flush=True)

    try:
        seg = Sam3Segmenter(
            weights=wpath,
            device=sam_dev,
            half=sam_half,
            conf=args.conf,
            imgsz=imgsz_kw,
            compile_graph=args.compile,
        )
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    cuda_info = "off"
    if torch.cuda.is_available():
        cuda_info = torch.cuda.get_device_name(0)
    print(
        f"SAM3: device={seg.device} fp16={seg.use_fp16} imgsz={imgsz_kw or 'default'} "
        f"({imgsz_note}) compile={bool(args.compile)} | CUDA: {cuda_info}",
        flush=True,
    )
    if sam_dev == "cpu" and args.imgsz >= 800:
        print(
            "提示: CPU + 大 imgsz 极慢，可试 未指定 --imgsz（自动 640）、--auto-cpu-imgsz 512 或 --stride 2",
            flush=True,
        )

    tb = parse_box(args.target_bbox.strip() or None)
    bbox_tgt = None
    if tb is not None:
        bbox_tgt = [[float(tb[0]), float(tb[1]), float(tb[2]), float(tb[3])]]
    text_list = None if bbox_tgt else [args.text.strip() or "object"]

    if not args.no_warmup:
        try:
            if bbox_tgt:
                seg.warmup(text_list=None, bboxes_xyxy=bbox_tgt)
            else:
                seg.warmup(text_list=text_list or ["object"], bboxes_xyxy=None)
        except Exception as ex:
            print(f"预热跳过: {ex}", flush=True)

    camera = LxCamera(lib)
    camera.DcSetInfoOutput(2, False, "./", 0)
    state, _dl, dev_num = camera.DcGetDeviceList()
    R.check(camera, state, "DcGetDeviceList")
    if dev_num <= 0 and args.open_mode == "index":
        print("未枚举到设备，无法按索引打开", file=sys.stderr)
        return 1

    mode_map = {
        "index": LX_OPEN_MODE.OPEN_BY_INDEX,
        "ip": LX_OPEN_MODE.OPEN_BY_IP,
        "sn": LX_OPEN_MODE.OPEN_BY_SN,
        "id": LX_OPEN_MODE.OPEN_BY_ID,
    }
    state, handle, dev_info = camera.DcOpenDevice(mode_map[args.open_mode], args.param)
    R.check(camera, state, "DcOpenDevice")
    print(f"设备: {dev_info.ip} sn={dev_info.sn}", flush=True)

    vis = None
    o3d_geom = None
    if args.o3d:
        vis, o3d_geom = _try_open3d_vis()
        if vis is not None:
            print("已打开 Open3D 点云窗口", flush=True)
        else:
            print("Open3D 不可用或创建窗口失败", flush=True)

    frame_i = 0
    last_t = time.time()
    fps_smooth = 0.0
    last_sam_ms = 0.0
    last_masks: list | None = None

    try:
        R.configure_rgbd_alignment(camera, handle)
        R.check(camera, camera.DcStartStream(handle), "DcStartStream")

        win = "rgbd_sam3_live"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        while True:
            rgb, xyz, st = R.grab_rgb_xyz(camera, handle)
            if st != LX_STATE.LX_SUCCESS or rgb is None:
                rec = getattr(LX_STATE, "LX_E_RECONNECTING", None)
                if rec is not None and st == rec:
                    continue
                continue
            if xyz is None:
                continue

            H, W = rgb.shape[:2]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            run_sam = (frame_i % max(1, args.stride) == 0) or last_masks is None
            if run_sam:
                t_sam = time.perf_counter()
                if bbox_tgt:
                    masks, _wh = seg.predict_masks_u8(
                        bgr, text_list=None, bboxes_xyxy=bbox_tgt
                    )
                else:
                    masks, _wh = seg.predict_masks_u8(
                        bgr, text_list=text_list, bboxes_xyxy=None
                    )
                last_sam_ms = (time.perf_counter() - t_sam) * 1000.0
                last_masks = masks

            mask_u8 = _merge_masks(last_masks or [], args.instance)
            if mask_u8 is None:
                overlay = bgr.copy()
                cv2.putText(
                    overlay,
                    "no mask",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow(win, overlay)
                frame_i += 1
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                continue

            filt = PC.filter_organized_xyz_with_mask(
                xyz,
                mask_u8,
                erode_iters=args.erode_iters,
                erode_kernel=args.erode_kernel,
                no_depth_filter=args.no_depth_filter,
                z_mad_k=args.z_mad_k,
                z_min_band_mm=args.z_min_band_mm,
                z_max_band_mm=args.z_max_band_mm,
                z_filter_passes=args.z_filter_passes,
            )
            cent = PC.centroid_xyz(filt)
            npts = len(filt)

            # 2D 叠加
            mcol = np.zeros_like(bgr)
            mcol[:, :, 1] = mask_u8
            overlay = cv2.addWeighted(bgr, 0.65, mcol, 0.35, 0)
            line1 = f"pts={npts}"
            line2 = (
                f"cx,cy,cz=({cent[0]:.1f},{cent[1]:.1f},{cent[2]:.1f})"
                if cent
                else "centroid=N/A"
            )
            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 1e-6:
                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt)
            line0 = (
                f"fps~{fps_smooth:.1f} stride={args.stride} "
                f"sam~{last_sam_ms:.0f}ms"
            )
            cv2.putText(overlay, line0, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(overlay, line1, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(overlay, line2, (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(
                overlay,
                "q/Esc quit",
                (10, H - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )
            cv2.imshow(win, overlay)

            if vis is not None and o3d_geom is not None and npts > 0:
                import open3d as o3d

                pts_d = filt.astype(np.float64)
                if npts > 80000:
                    idx = np.random.choice(npts, 80000, replace=False)
                    pts_d = pts_d[idx]
                o3d_geom.points = o3d.utility.Vector3dVector(pts_d)
                vis.update_geometry(o3d_geom)
                vis.poll_events()
                vis.update_renderer()

            frame_i += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

        cv2.destroyWindow(win)
        if vis is not None:
            vis.destroy_window()
    finally:
        try:
            camera.DcStopStream(handle)
        except Exception:
            pass
        try:
            camera.DcCloseDevice(handle)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
