#!/usr/bin/env python3
"""
使用 Ultralytics SAM3（本地 sam3.pt）做概念分割：文本提示或目标图上的框提示。

参考: https://docs.ultralytics.com/models/sam-3/

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
    run_sam3_semantic,
    save_from_results,
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
        help="如 cpu 或 0；默认自动",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="FP16（GPU 上更快；CPU 可不加）",
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

    device_str = args.device
    if not device_str:
        device_str = "cpu"
        if torch.cuda.is_available():
            device_str = "0"

    use_half = args.half and torch.cuda.is_available()

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
                concept = caption_exemplar(img, "cuda" if torch.cuda.is_available() else "cpu")
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
    print(f"device={device_str} half={use_half}", flush=True)

    try:
        results = run_sam3_semantic(
            target_path,
            str(wpath),
            device_str,
            use_half,
            args.conf,
            text_list,
            bbox_tgt,
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
