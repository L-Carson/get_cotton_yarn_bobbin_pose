#!/usr/bin/env python3
"""
用 RGB 上的二值分割 mask，从 LxCamera 导出的稀疏点云（.pcd，仅非零点）中筛出对应 3D 点并保存。

实现见包 mrdvs.pc_mask；本脚本仅提供命令行入口，便于单独调试。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from mrdvs import pc_mask as PC


def main() -> int:
    import cv2

    p = argparse.ArgumentParser(
        description="用 SAM 等分割 mask 从稀疏点云 pcd 中筛选并保存",
    )
    p.add_argument("--rgb", required=True, help="与点云同帧的 RGB（定分辨率）")
    p.add_argument("--mask", required=True, help="单通道 mask PNG（白=保留）")
    p.add_argument("--out", "-o", required=True, help="输出 .pcd 路径")
    p.add_argument(
        "--pcd",
        default="",
        help="稀疏点云 .pcd（与 DcSaveXYZ 的 pcd）",
    )
    p.add_argument(
        "--dense-txt",
        default="",
        help="若改用 DcSaveXYZ 保存的 .txt（按图像顺序完整网格），填此路径并配合 --grid-w --grid-h",
    )
    p.add_argument("--grid-w", type=int, default=0, help="3D 图宽度（仅 dense-txt）")
    p.add_argument("--grid-h", type=int, default=0, help="3D 图高度（仅 dense-txt）")
    p.add_argument(
        "--intrinsics-json",
        default="",
        help='JSON: {"fx","fy","cx","cy"}，若指定则覆盖下面四个参数',
    )
    p.add_argument("--fx", type=float, default=float("nan"))
    p.add_argument("--fy", type=float, default=float("nan"))
    p.add_argument("--cx", type=float, default=float("nan"))
    p.add_argument("--cy", type=float, default=float("nan"))
    p.add_argument(
        "--flip-v",
        action="store_true",
        help="投影后翻转 v（若上下颠倒可试）",
    )
    p.add_argument(
        "--z-min",
        type=float,
        default=1e-3,
        help="忽略 |Z| 小于该值的点（投影分母）",
    )
    p.add_argument(
        "--erode-iters",
        type=int,
        default=3,
        help="分割 mask 腐蚀迭代次数，0=不腐蚀；默认缩小前景减轻边缘误点",
    )
    p.add_argument(
        "--erode-kernel",
        type=int,
        default=5,
        help="腐蚀结构元素直径（奇数，≥3）",
    )
    p.add_argument(
        "--no-depth-filter",
        action="store_true",
        help="关闭深度 MAD 离群剔除（仅 mask+投影）",
    )
    p.add_argument(
        "--z-mad-k",
        type=float,
        default=3.5,
        help="深度 MAD 倍数，越大越宽松（默认 3.5）",
    )
    p.add_argument(
        "--z-min-band-mm",
        type=float,
        default=25.0,
        help="深度带宽下限（mm），避免物体很薄时 band 过小",
    )
    p.add_argument(
        "--z-max-band-mm",
        type=float,
        default=0.0,
        help="深度带宽上限（mm），0=不限制；若主体薄但仍误留远处点可设如 200",
    )
    p.add_argument(
        "--z-filter-passes",
        type=int,
        default=2,
        help="深度剔除迭代遍数（默认 2），可收紧残留离群点",
    )
    p.add_argument(
        "--centroid-out",
        default="",
        help="可选：将重心 x y z 写入该文本文件（一行三个数，空格分隔）",
    )
    args = p.parse_args()

    ij = (args.intrinsics_json or "").strip()
    if ij:
        with open(os.path.abspath(ij), encoding="utf-8") as f:
            j = json.load(f)
        args.fx = float(j["fx"])
        args.fy = float(j["fy"])
        args.cx = float(j["cx"])
        args.cy = float(j["cy"])

    use_dense = bool(args.dense_txt.strip())
    use_pcd = bool(args.pcd.strip())
    if use_dense == use_pcd:
        print("请二选一: --pcd 或 --dense-txt", file=sys.stderr)
        return 1

    rgb = cv2.imread(os.path.abspath(args.rgb))
    if rgb is None:
        print(f"无法读取 RGB: {args.rgb}", file=sys.stderr)
        return 1
    H, W = rgb.shape[:2]

    mpath = os.path.abspath(args.mask)
    mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"无法读取 mask: {mpath}", file=sys.stderr)
        return 1
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    mask = PC.erode_mask(mask, args.erode_iters, args.erode_kernel)

    if use_dense:
        if args.grid_w <= 0 or args.grid_h <= 0:
            print("dense-txt 需要 --grid-w / --grid-h", file=sys.stderr)
            return 1
        filtered = PC.dense_txt_filter(
            os.path.abspath(args.dense_txt), mask, args.grid_w, args.grid_h
        )
    else:
        if any(
            x != x
            for x in (args.fx, args.fy, args.cx, args.cy)  # nan check
        ):
            print(
                "稀疏 pcd 需要相机内参: --fx --fy --cx --cy（与 RGB 分辨率一致）",
                file=sys.stderr,
            )
            return 1
        pts = PC.load_ascii_pcd_xyz(os.path.abspath(args.pcd))
        filtered = PC.project_and_filter(
            pts,
            mask,
            args.fx,
            args.fy,
            args.cx,
            args.cy,
            args.z_min,
            args.flip_v,
            (W, H),
        )

    n_before_depth = len(filtered)
    if not args.no_depth_filter and n_before_depth > 0:
        filtered = PC.filter_depth_outliers(
            filtered,
            args.z_mad_k,
            args.z_min_band_mm,
            args.z_max_band_mm,
            args.z_filter_passes,
        )

    n = len(filtered)
    out_path = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    PC.save_pcd_ascii(out_path, filtered)
    if args.no_depth_filter:
        print(f"已保存 {n} 点 -> {out_path}（未做深度离群剔除）", flush=True)
    else:
        print(
            f"已保存 {n} 点 -> {out_path}（mask 腐蚀 iters={args.erode_iters}；"
            f"深度剔除前 {n_before_depth} 点）",
            flush=True,
        )

    cent = PC.centroid_xyz(filtered)
    if cent is not None:
        cx, cy, cz = cent
        print(
            f"重心 (算术平均, 相机坐标系): "
            f"x={cx:.6f}  y={cy:.6f}  z={cz:.6f}  (单位与点云一致, 一般为 mm)",
            flush=True,
        )
        co = (args.centroid_out or "").strip()
        if co:
            cpath = os.path.abspath(co)
            cdir = os.path.dirname(cpath)
            if cdir:
                os.makedirs(cdir, exist_ok=True)
            with open(cpath, "w", encoding="utf-8") as f:
                f.write(f"{cx:.8f} {cy:.8f} {cz:.8f}\n")
            print(f"重心已写入: {cpath}", flush=True)
    else:
        print("重心: 无有效点，无法计算", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
