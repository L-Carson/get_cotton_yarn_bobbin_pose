#!/usr/bin/env python3
"""
使用 Ultralytics SAM3（本地 sam3.pt）做概念分割：文本提示或目标图上的框提示。

参考: https://docs.ultralytics.com/models/sam-3/

默认权重路径（与脚本同目录）:
  /home/ly/cursor/mrdvs/get_object/sam3.pt

依赖:
  pip install -U "ultralytics>=8.3.237" opencv-python pillow numpy torch
若提示 CLIP 错误: pip uninstall clip -y && pip install git+https://github.com/ultralytics/CLIP.git

说明:
  - SAM3（Ultralytics）对「目标图」分割时，有效提示是：① 英文 --text；② 在「目标图」像素坐标下的 --target-bbox。
    不能把「仅图例上的框」自动用到目标图；图例一般配合 BLIP 生成英文概念，再用同一概念在目标图上做文本分割。
  - 未传 --text 时会尝试用 BLIP 下载/加载模型；若环境配置了 SOCKS 代理且未装 socksio，可能报错，可临时取消代理或: pip install "httpx[socks]"
  - --ref-box 仅裁剪参考图供 BLIP 看局部，不直接参与 SAM3 几何提示。

用法:cotton yarn bobbin
  python3 sam3_exemplar_segment.py -e ./rgb_000001.png -t ./rgb_000002.png -o ./sam3_out --text "hand"
  python3 sam3_exemplar_segment.py -t ./rgb_000002.png -o ./out --text "hand"   # 仅需目标图
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_WEIGHTS = _SCRIPT_DIR / "sam3.pt"


def _strip_proxy_env() -> None:
    """避免 BLIP/HuggingFace 下载权重时走 SOCKS 却未安装 socksio 导致失败。"""
    for k in list(os.environ.keys()):
        if "proxy" in k.lower():
            del os.environ[k]


def _parse_box(s: str | None) -> tuple[int, int, int, int] | None:
    if not s:
        return None
    parts = s.replace(",", " ").split()
    if len(parts) != 4:
        raise ValueError("框需要 4 个数: x1 y1 x2 y2")
    return tuple(int(x) for x in parts)


def _caption_exemplar(pil_img, device: str) -> str:
    _strip_proxy_env()
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch

    mid = "Salesforce/blip-image-captioning-base"
    proc = BlipProcessor.from_pretrained(mid)
    mdl = BlipForConditionalGeneration.from_pretrained(mid).to(device)
    mdl.eval()
    inputs = proc(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_length=48, num_beams=4)
    return proc.decode(out[0], skip_special_tokens=True).strip()


def _run_sam3_semantic(
    target_path: str,
    weights: str,
    device_str: str,
    half: bool,
    conf: float,
    text_list: list[str] | None,
    bboxes_xyxy: list[list[float]] | None,
):
    from ultralytics.models.sam import SAM3SemanticPredictor

    overrides = dict(
        conf=conf,
        iou=0.5,
        task="segment",
        mode="predict",
        model=weights,
        half=half,
        save=False,
        verbose=False,
    )
    if device_str:
        overrides["device"] = device_str

    predictor = SAM3SemanticPredictor(overrides=overrides)
    predictor.set_image(target_path)

    if bboxes_xyxy is not None and len(bboxes_xyxy) > 0:
        return predictor(bboxes=bboxes_xyxy)
    if text_list:
        return predictor(text=text_list)
    raise ValueError("需要 --text 或 --target-bbox")


def _save_from_results(results, target_path: str, out_dir: str, base: str) -> int:
    import cv2
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)
    res = results[0]
    img_bgr = cv2.imread(target_path)
    if img_bgr is None:
        raise RuntimeError(f"无法读取图像: {target_path}")
    h, w = img_bgr.shape[:2]

    if res.masks is None or len(res.masks) == 0:
        print("未检测到实例。", flush=True)
        return 0

    masks_t = res.masks.data.cpu().numpy()
    n = masks_t.shape[0]
    overlay = img_bgr.copy().astype(np.float32)

    for i in range(n):
        m = masks_t[i]
        if m.shape[:2] != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
        else:
            m = m > 0.5
        mask_u8 = (m.astype(np.uint8)) * 255
        cv2.imwrite(os.path.join(out_dir, f"{base}_mask_{i:02d}.png"), mask_u8)
        col = (0, 255, 0)
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_u8 > 0,
                0.5 * overlay[:, :, c] + 0.5 * col[c],
                overlay[:, :, c],
            )
        print(f"  实例 {i}", flush=True)

    cv2.imwrite(os.path.join(out_dir, f"{base}_overlay.png"), overlay.astype(np.uint8))
    return n


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
        default=str(_DEFAULT_WEIGHTS),
        help=f"sam3.pt 路径，默认: {_DEFAULT_WEIGHTS}",
    )
    parser.add_argument(
        "--text",
        default="",
        help="英文概念，如 hand / red mug；不设则尝试 BLIP（需 --exemplar）或 --target-bbox",
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

    tb = _parse_box(args.target_bbox.strip() or None)
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
            rb = _parse_box(args.ref_box.strip() or None)
            if rb is not None:
                img = img.crop(rb)
                print(f"已按 --ref-box 裁剪参考图: {rb}")
            print("BLIP 配文…", flush=True)
            try:
                concept = _caption_exemplar(img, "cuda" if torch.cuda.is_available() else "cpu")
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
        results = _run_sam3_semantic(
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

    n = _save_from_results(results, target_path, out_dir, base)
    print(f"完成: {n} 个掩码 -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
