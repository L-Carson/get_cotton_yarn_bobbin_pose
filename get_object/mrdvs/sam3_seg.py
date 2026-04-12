"""Ultralytics SAM3：文本/框分割；支持文件路径或 BGR ndarray。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = _SCRIPT_DIR / "sam3.pt"


def strip_proxy_env() -> None:
    for k in list(os.environ.keys()):
        if "proxy" in k.lower():
            del os.environ[k]


def parse_box(s: str | None) -> tuple[int, int, int, int] | None:
    if not s:
        return None
    parts = s.replace(",", " ").split()
    if len(parts) != 4:
        raise ValueError("框需要 4 个数: x1 y1 x2 y2")
    return tuple(int(x) for x in parts)


def caption_exemplar(pil_img, device: str) -> str:
    strip_proxy_env()
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


def run_sam3_semantic(
    target_source: str | object,
    weights: str,
    device_str: str,
    half: bool,
    conf: float,
    text_list: list[str] | None,
    bboxes_xyxy: list[list[float]] | None,
):
    """target_source: 图像路径(str) 或 BGR ndarray。"""
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
    predictor.set_image(target_source)

    if bboxes_xyxy is not None and len(bboxes_xyxy) > 0:
        return predictor(bboxes=bboxes_xyxy)
    if text_list:
        return predictor(text=text_list)
    raise ValueError("需要 text_list 或 bboxes_xyxy")


def results_to_mask_u8_list(results, h: int, w: int):
    """返回每张 mask 的 uint8(H,W) 0/255。"""
    import cv2
    import numpy as np

    res = results[0]
    if res.masks is None or len(res.masks) == 0:
        return []

    masks_t = res.masks.data.cpu().numpy()
    n = masks_t.shape[0]
    out: list = []
    for i in range(n):
        m = masks_t[i]
        if m.shape[:2] != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
        else:
            m = m > 0.5
        out.append((m.astype(np.uint8)) * 255)
    return out


def save_from_results(results, target_path: str, out_dir: str, base: str) -> int:
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


class Sam3Segmenter:
    """封装单次加载权重的 SAM3，对 ndarray 或路径推理。"""

    def __init__(
        self,
        weights: str | Path | None = None,
        device: str = "",
        half: bool = False,
        conf: float = 0.25,
    ):
        import torch

        self._weights = str(Path(weights or DEFAULT_WEIGHTS).expanduser().resolve())
        if not Path(self._weights).is_file():
            raise FileNotFoundError(f"未找到权重: {self._weights}")

        if not device:
            device = "cpu"
            if torch.cuda.is_available():
                device = "0"
        self._device = device
        self._half = bool(half and torch.cuda.is_available())
        self._conf = conf
        self._predictor = None

    def _ensure_predictor(self):
        if self._predictor is not None:
            return self._predictor
        from ultralytics.models.sam import SAM3SemanticPredictor

        overrides = dict(
            conf=self._conf,
            iou=0.5,
            task="segment",
            mode="predict",
            model=self._weights,
            half=self._half,
            save=False,
            verbose=False,
            device=self._device,
        )
        self._predictor = SAM3SemanticPredictor(overrides=overrides)
        return self._predictor

    def predict(
        self,
        image_bgr,
        text_list: list[str] | None = None,
        bboxes_xyxy: list[list[float]] | None = None,
    ):
        """
        image_bgr: 路径 str 或 BGR uint8 ndarray。
        """
        pred = self._ensure_predictor()
        pred.set_image(image_bgr)
        if bboxes_xyxy is not None and len(bboxes_xyxy) > 0:
            return pred(bboxes=bboxes_xyxy)
        if text_list:
            return pred(text=text_list)
        raise ValueError("需要 text_list 或 bboxes_xyxy")

    def predict_masks_u8(
        self,
        image_bgr,
        text_list: list[str] | None = None,
        bboxes_xyxy: list[list[float]] | None = None,
    ):
        """返回 list[np.ndarray] 单通道 mask 0/255，以及 (H,W)。"""
        import numpy as np

        if isinstance(image_bgr, str):
            import cv2

            im = cv2.imread(image_bgr)
            if im is None:
                raise RuntimeError(f"无法读取: {image_bgr}")
            h, w = im.shape[:2]
        else:
            im = np.asarray(image_bgr)
            h, w = im.shape[:2]

        results = self.predict(im, text_list=text_list, bboxes_xyxy=bboxes_xyxy)
        return results_to_mask_u8_list(results, h, w), (h, w)
