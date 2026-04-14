"""Ultralytics SAM3：文本/框分割；支持文件路径或 BGR ndarray。

官方环境（SAM3 / Ultralytics，GPU 推理）：Python ≥3.12、PyTorch ≥2.7、
NVIDIA GPU 且使用 **CUDA 12.6+** 对应的 PyTorch 轮子（如 cu126，以 pytorch.org 为准）。
驱动需不低于该 CUDA 版本对驱动的最低要求。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = _SCRIPT_DIR / "sam3.pt"

_GPU_TUNED = False


def configure_pytorch_gpu() -> None:
    """在首次使用 GPU 推理前调用：cuDNN benchmark、TF32、矩阵乘精度（Tensor Core）。"""
    global _GPU_TUNED
    if _GPU_TUNED:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            _GPU_TUNED = True
            return
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, "matmul") and hasattr(
            torch.backends.cuda.matmul, "allow_tf32"
        ):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        # Ampere+ 上加速 FP32 matmul（与 half 推理可并存）
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        _GPU_TUNED = True
    except Exception:
        _GPU_TUNED = True


def build_sam3_overrides(
    *,
    weights: str,
    device: str,
    half: bool,
    conf: float,
    imgsz: int | None = None,
    compile_graph: bool | str = False,
) -> dict:
    """Ultralytics SAM3SemanticPredictor 的 overrides 公共构造。"""
    o = dict(
        conf=conf,
        iou=0.5,
        task="segment",
        mode="predict",
        model=weights,
        half=half,
        save=False,
        verbose=False,
        device=device,
    )
    if imgsz is not None:
        o["imgsz"] = int(imgsz)
    if compile_graph:
        o["compile"] = compile_graph if isinstance(compile_graph, str) else True
    return o


def resolve_device(device: str = "") -> str:
    """空字符串：有 CUDA 用 ``0``，否则 ``cpu``。"""
    import torch

    d = (device or "").strip()
    if d:
        return d
    return "0" if torch.cuda.is_available() else "cpu"


def explain_cuda_unavailable_if_needed() -> None:
    """在 import torch 之后调用：说明为何未用 GPU（驱动/PyTorch 不匹配等）。"""
    import torch

    if torch.cuda.is_available():
        return
    lines = [
        "[SAM3] 当前未启用 CUDA，将使用 CPU 推理（SAM3 在 CPU 上往往每帧数秒，属正常）。",
        "  SAM3 建议环境: Python≥3.12, PyTorch≥2.7, GPU 需 CUDA 12.6+ 的 PyTorch 构建（见 https://pytorch.org）。",
    ]
    cv = getattr(torch.version, "cuda", None)
    if cv:
        lines.append(
            f"  PyTorch 为 CUDA 构建 (torch.version.cuda={cv})，但 cuda.is_available()=False，"
            "多为显卡驱动版本低于当前 PyTorch 所需的 CUDA 运行时。"
        )
        lines.append(
            "  处理: ① 升级 NVIDIA 驱动至支持所用 CUDA 版本；② 或重装与驱动匹配的 torch（GPU 推荐 cu126 等与 CUDA≥12.6 对应的轮子）。"
        )
    else:
        lines.append(
            "  当前为 CPU 版 PyTorch；若需 GPU 请安装带 CUDA 12.6+ 的 torch（PyTorch≥2.7, Python≥3.12）。"
        )
    print("\n".join(lines), flush=True)


def resolve_imgsz_for_runtime(
    user_imgsz: int,
    device_str: str,
    *,
    auto_cpu_imgsz: int = 640,
    no_auto: bool = False,
) -> tuple[int | None, str]:
    """
    返回 (传给 Ultralytics 的 imgsz, 说明字符串)。
    user_imgsz>0: 用户指定；=0 且 CPU 且非 no_auto: 使用 auto_cpu_imgsz；否则 None（Ultralytics 默认）。
    """
    import torch

    if user_imgsz > 0:
        return user_imgsz, f"user={user_imgsz}"
    use_cpu = device_str.strip() == "cpu" or not torch.cuda.is_available()
    if use_cpu and not no_auto:
        return auto_cpu_imgsz, f"auto_cpu={auto_cpu_imgsz}"
    return None, "ultralytics_default"


def resolve_half(fp16: bool | None, device_str: str) -> bool:
    """fp16 is None：仅在 CUDA 上默认开 FP16。"""
    import torch

    if fp16 is not None:
        return bool(fp16) and torch.cuda.is_available()
    return torch.cuda.is_available() and device_str not in ("cpu",)


def ultralytics_device_to_torch(device_str: str) -> str:
    """供 transformers 等使用：``0`` -> ``cuda:0``。"""
    s = device_str.strip()
    if s == "cpu" or not s:
        return "cpu"
    if s.isdigit():
        return f"cuda:{s}"
    if s.startswith("cuda"):
        return s
    return s


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
    *,
    imgsz: int | None = None,
    compile_graph: bool | str = False,
):
    """target_source: 图像路径(str) 或 BGR ndarray。"""
    import torch
    from ultralytics.models.sam import SAM3SemanticPredictor

    ds = (device_str or "").strip()
    dev = resolve_device("") if not ds else ds
    if dev != "cpu":
        configure_pytorch_gpu()

    overrides = build_sam3_overrides(
        weights=weights,
        device=dev,
        half=half,
        conf=conf,
        imgsz=imgsz,
        compile_graph=compile_graph,
    )

    predictor = SAM3SemanticPredictor(overrides=overrides)
    with torch.inference_mode():
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
        half: bool | None = None,
        conf: float = 0.25,
        *,
        imgsz: int | None = None,
        compile_graph: bool | str = False,
    ):
        import torch

        self._weights = str(Path(weights or DEFAULT_WEIGHTS).expanduser().resolve())
        if not Path(self._weights).is_file():
            raise FileNotFoundError(f"未找到权重: {self._weights}")

        ds = (device or "").strip()
        self._device = resolve_device("") if not ds else ds
        if self._device != "cpu":
            configure_pytorch_gpu()
        self._half = resolve_half(half, self._device)
        self._conf = conf
        self._imgsz = imgsz
        self._compile_graph = compile_graph
        self._predictor = None

    def _ensure_predictor(self):
        if self._predictor is not None:
            return self._predictor
        from ultralytics.models.sam import SAM3SemanticPredictor

        overrides = build_sam3_overrides(
            weights=self._weights,
            device=self._device,
            half=self._half,
            conf=self._conf,
            imgsz=self._imgsz,
            compile_graph=self._compile_graph,
        )
        self._predictor = SAM3SemanticPredictor(overrides=overrides)
        return self._predictor

    @property
    def device(self) -> str:
        return self._device

    @property
    def use_fp16(self) -> bool:
        return self._half

    def warmup(
        self,
        text_list: list[str] | None = None,
        bboxes_xyxy: list[list[float]] | None = None,
    ) -> None:
        """一次空图推理，预热 CUDA/cuDNN/torch.compile，减轻首帧卡顿。"""
        import numpy as np

        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.predict(
            dummy,
            text_list=text_list or ["object"],
            bboxes_xyxy=bboxes_xyxy,
        )

    def predict(
        self,
        image_bgr,
        text_list: list[str] | None = None,
        bboxes_xyxy: list[list[float]] | None = None,
    ):
        """
        image_bgr: 路径 str 或 BGR uint8 ndarray。
        """
        import torch

        pred = self._ensure_predictor()
        with torch.inference_mode():
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
