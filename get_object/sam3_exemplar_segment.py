#!/usr/bin/env python3
"""
使用 Ultralytics SAM3（本地 sam3.pt）做概念分割：文本提示或目标图上的框提示。

参考: https://docs.ultralytics.com/models/sam-3/

环境（GPU 推理，与 Ultralytics SAM3 一致）:
  Python ≥3.12，PyTorch ≥2.7，NVIDIA GPU + 与驱动匹配的 **CUDA 12.6+** PyTorch 构建
  （安装见 https://pytorch.org ，选择 CUDA 12.6 或更新对应的 wheel）。

实现见 mrdvs.sam3_seg；本脚本为命令行入口，便于单独调试。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from mrdvs.sam3_seg import (
    DEFAULT_WEIGHTS,
    caption_exemplar,
    parse_box,
    resolve_device,
    resolve_half,
    run_sam3_semantic,
    save_from_results,
    ultralytics_device_to_torch,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ultralytics SAM3：本地 sam3.pt，文本或框提示分割目标图",
    )
    parser.add_argument("--exemplar", "-e", default="", help="参考图（可选，用于 BLIP 配文）")
    parser.add_argument("--target", "-t", required=True, help="待分割的目标图")
    parser.add_argument("--out", "-o", default="sam3_output", help="输出目录")
    parser.add_argument(
        "--weights",
        "-w",
        default=str(DEFAULT_WEIGHTS),
        help=f"sam3.pt 路径，默认: {DEFAULT_WEIGHTS}",
    )
    parser.add_argument(
        "--text",
        default="",
        help="英文概念，如 hand / cotton yarn bobbin；不设则尝试 BLIP（需 --exemplar）或 --target-bbox",
    )
    parser.add_argument(
        "--ref-box",
        default="",
        help="参考图上裁剪区域 x1 y1 x2 y2，仅配合 BLIP",
    )
    parser.add_argument(
        "--target-bbox",
        default="",
        help="目标图上的框 x1 y1 x2 y2（像素），用 exemplar 式框提示分割，无需文本",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument(
        "--device",
        default="",
        help="如 cpu、0；空则自动选 GPU（有 CUDA 时）",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="强制 CPU",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="GPU 上使用 FP32（默认 FP16 加速）",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=0,
        help="SAM 正方形边长，0=使用 Ultralytics 默认；可试 896/736 降低算量（可能略损精度）",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="启用 torch.compile（PyTorch 2+，首次运行会较慢编译）",
    )
    args = parser.parse_args()

    wpath = Path(args.weights).expanduser().resolve()
    if not wpath.is_file():
        print(f"未找到权重文件: {wpath}", file=sys.stderr)
        print("请将 sam3.pt 放在本目录或传入 --weights", file=sys.stderr)
        return 1

    target_path = os.path.abspath(args.target)
    if not os.path.isfile(target_path):
        print(f"找不到目标图: {target_path}", file=sys.stderr)
        return 1

    try:
        from PIL import Image
    except ImportError:
        print("需要 pillow: pip install pillow", file=sys.stderr)
        return 1

    import torch

    if args.cpu:
        device_str = "cpu"
    else:
        device_str = (args.device or "").strip() or resolve_device("")
    use_half = resolve_half(False if args.fp32 else None, device_str)

    text_list: list[str] | None = None
    bbox_tgt: list[list[float]] | None = None

    tb = parse_box(args.target_bbox.strip() or None)
    if tb is not None:
        bbox_tgt = [[float(tb[0]), float(tb[1]), float(tb[2]), float(tb[3])]]

    if bbox_tgt is None:
        concept = (args.text or "").strip()
        if not concept:
            ex = (args.exemplar or "").strip()
            if not ex:
                print(
                    "请提供以下之一: --text；或 --target-bbox；或 --exemplar（将用 BLIP 配文）",
                    file=sys.stderr,
                )
                return 1
            ex_path = os.path.abspath(ex)
            if not os.path.isfile(ex_path):
                print(f"找不到参考图: {ex_path}", file=sys.stderr)
                return 1
            img = Image.open(ex_path).convert("RGB")
            rb = parse_box(args.ref_box.strip() or None)
            if rb is not None:
                img = img.crop(rb)
                print(f"已按 --ref-box 裁剪参考图: {rb}")
            print("BLIP 配文…", flush=True)
            try:
                concept = caption_exemplar(
                    img, ultralytics_device_to_torch(device_str)
                )
            except Exception as e:
                print(
                    f"BLIP 失败: {e}\n"
                    "可改用: --text \"英文物体名\"；或安装 pip install \"httpx[socks]\"；"
                    "或临时取消终端代理后重试。",
                    file=sys.stderr,
                )
                return 1
            print(f"自动概念: {concept!r}", flush=True)
        text_list = [concept]

    print(f"权重: {wpath}", flush=True)
    print(
        f"device={device_str} fp16={use_half} | "
        f"cuda={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}",
        flush=True,
    )

    imgsz_kw = None if args.imgsz <= 0 else args.imgsz
    try:
        results = run_sam3_semantic(
            target_path,
            str(wpath),
            device_str,
            use_half,
            args.conf,
            text_list,
            bbox_tgt,
            imgsz=imgsz_kw,
            compile_graph=args.compile,
        )
    except Exception as e:
        print(
            "SAM3 推理失败。若缺依赖请: pip install -U ultralytics timm\n"
            "CLIP: pip install git+https://github.com/ultralytics/CLIP.git\n"
            f"原因: {e}",
            file=sys.stderr,
        )
        return 1

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(target_path))[0]
    if text_list:
        with open(os.path.join(out_dir, f"{base}_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(text_list[0] + "\n")

    n = save_from_results(results, target_path, out_dir, base)
    print(f"完成: {n} 个掩码 -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
