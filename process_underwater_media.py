#!/usr/bin/env python3
"""Batch underwater enhancement for images and videos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhance underwater JPG images and MOV videos with this repo's models."
    )
    parser.add_argument("--input", required=True, help="Input file or directory.")
    parser.add_argument("--output", required=True, help="Directory for enhanced media.")
    parser.add_argument(
        "--engine",
        default="repo",
        choices=["repo", "heuristic"],
        help="Use the repo model ('repo') or a Pillow-only fallback ('heuristic').",
    )
    parser.add_argument(
        "--checkpoint",
        help="Path to a model checkpoint (.pth). Required for --engine repo.",
    )
    parser.add_argument(
        "--model",
        default="UWCNN",
        choices=["UWCNN", "UIEC2Net"],
        help="Model architecture that matches the checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--video-codec",
        default="mp4v",
        help="OpenCV fourcc codec for processed videos.",
    )
    parser.add_argument(
        "--video-ext",
        default=".mp4",
        help="Output suffix for processed videos.",
    )
    return parser.parse_args()


def require_runtime():
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required to run this repo's enhancement models, but it is not installed."
        ) from exc


def check_repo_requirements():
    missing = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    if missing:
        raise SystemExit(
            "The repo model engine requires these missing dependencies: "
            + ", ".join(missing)
            + ". Use --engine heuristic for image-only processing without extra installs."
        )


def get_device(device_arg: str):
    import torch

    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name: str, checkpoint_path: Path, device):
    import torch
    from core.Models.UWModels import UIEC2Net, UWCNN

    constructors = {
        "UWCNN": UWCNN,
        "UIEC2Net": UIEC2Net,
    }
    model = constructors[model_name](get_parameter=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    state_dict = {
        key.split("module.", 1)[-1] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def model_uses_normalization(model_name: str) -> bool:
    return model_name == "UWCNN"


def media_files(input_path: Path) -> Iterable[Path]:
    supported = {".jpg", ".jpeg", ".png", ".mov", ".mp4", ".avi"}
    if input_path.is_file():
        if input_path.suffix.lower() in supported:
            yield input_path
        return

    for path in sorted(input_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in supported:
            yield path


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png"}


def is_video(path: Path) -> bool:
    return path.suffix.lower() in {".mov", ".mp4", ".avi"}


def pil_to_tensor(image: Image.Image, normalize: bool, device):
    import torch

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    if normalize:
        tensor = (tensor - 0.5) / 0.5
    return tensor.to(device)


def tensor_to_pil(output_tensor, normalize: bool) -> Image.Image:
    tensor = output_tensor.detach().cpu()[0]
    if normalize:
        tensor = ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)
    else:
        tensor = tensor.clamp(0.0, 1.0)
    array = tensor.permute(1, 2, 0).numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def enhance_image(image: Image.Image, model, normalize: bool, device) -> Image.Image:
    import torch

    rgb = ImageOps.exif_transpose(image).convert("RGB")
    inputs = pil_to_tensor(rgb, normalize=normalize, device=device)
    with torch.no_grad():
        outputs = model(inputs)
    return tensor_to_pil(outputs, normalize=normalize)


def simplest_color_balance(image: Image.Image, low_clip: float = 0.01, high_clip: float = 0.99) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    out = np.empty_like(array)
    for channel in range(3):
        plane = array[:, :, channel]
        low = np.quantile(plane, low_clip)
        high = np.quantile(plane, high_clip)
        if high <= low:
            out[:, :, channel] = plane
        else:
            out[:, :, channel] = np.clip((plane - low) * 255.0 / (high - low), 0, 255)
    return Image.fromarray(out.astype(np.uint8), mode="RGB")


def heuristic_underwater_enhance(image: Image.Image) -> Image.Image:
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)

    channel_means = arr.reshape(-1, 3).mean(axis=0)
    mean_gray = float(channel_means.mean())
    gains = mean_gray / np.maximum(channel_means, 1.0)
    gains[0] *= 1.15
    gains[2] *= 0.92

    balanced = np.clip(arr * gains.reshape(1, 1, 3), 0, 255).astype(np.uint8)
    out = Image.fromarray(balanced, mode="RGB")
    out = simplest_color_balance(out, low_clip=0.01, high_clip=0.99)
    out = ImageEnhance.Color(out).enhance(1.25)
    out = ImageEnhance.Contrast(out).enhance(1.12)
    out = ImageEnhance.Sharpness(out).enhance(1.08)
    out = out.filter(ImageFilter.MedianFilter(size=3))
    return out


def output_path_for(input_root: Path, input_file: Path, output_root: Path, video_ext: str) -> Path:
    relative = input_file.name if input_root.is_file() else str(input_file.relative_to(input_root))
    target = output_root / relative
    if is_video(input_file):
        target = target.with_suffix(video_ext)
    return target


def process_image_file(
    input_root: Path,
    input_file: Path,
    output_root: Path,
    model,
    normalize: bool,
    device,
    video_ext: str,
    engine: str,
):
    output_file = output_path_for(input_root, input_file, output_root, video_ext)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(input_file) as image:
        if engine == "repo":
            enhanced = enhance_image(image, model=model, normalize=normalize, device=device)
        else:
            enhanced = heuristic_underwater_enhance(image)
        save_kwargs = {"quality": 95} if output_file.suffix.lower() in {".jpg", ".jpeg"} else {}
        enhanced.save(output_file, **save_kwargs)
    print(f"saved image: {output_file}")


def process_video_file(
    input_root: Path,
    input_file: Path,
    output_root: Path,
    model,
    normalize: bool,
    device,
    video_ext: str,
    video_codec: str,
    engine: str,
):
    if engine != "repo":
        raise SystemExit(
            "Video processing is unavailable in this environment. The heuristic engine is image-only, "
            "and there is no OpenCV/ffmpeg installed for decoding .mov files."
        )
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "OpenCV is required for video processing, but it is not installed."
        ) from exc

    output_file = output_path_for(input_root, input_file, output_root, video_ext)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_file))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_file}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*video_codec[:4])
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output video for writing: {output_file}")

    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            enhanced = enhance_image(
                Image.fromarray(rgb, mode="RGB"),
                model=model,
                normalize=normalize,
                device=device,
            )
            bgr = cv2.cvtColor(np.asarray(enhanced), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            index += 1
            if frame_count > 0:
                print(f"processed video frame {index}/{frame_count}: {input_file.name}")
            else:
                print(f"processed video frame {index}: {input_file.name}")
    finally:
        capture.release()
        writer.release()

    print(f"saved video: {output_file}")


def main():
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")

    video_ext = args.video_ext if args.video_ext.startswith(".") else f".{args.video_ext}"
    device = None
    model = None
    normalize = False

    if args.engine == "repo":
        check_repo_requirements()
        require_runtime()
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required when --engine repo is used.")
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            raise SystemExit(f"Checkpoint does not exist: {checkpoint_path}")
        device = get_device(args.device)
        model = load_model(args.model, checkpoint_path, device)
        normalize = model_uses_normalization(args.model)

    files = list(media_files(input_path))
    if not files:
        raise SystemExit("No supported media files were found.")

    output_root.mkdir(parents=True, exist_ok=True)
    print(f"engine: {args.engine}")
    if args.engine == "repo":
        print(f"using model: {args.model}")
        print(f"device: {device}")
    print(f"items found: {len(files)}")

    for path in files:
        if is_image(path):
            process_image_file(
                input_root=input_path,
                input_file=path,
                output_root=output_root,
                model=model,
                normalize=normalize,
                device=device,
                video_ext=video_ext,
                engine=args.engine,
            )
        elif is_video(path):
            process_video_file(
                input_root=input_path,
                input_file=path,
                output_root=output_root,
                model=model,
                normalize=normalize,
                device=device,
                video_ext=video_ext,
                video_codec=args.video_codec,
                engine=args.engine,
            )


if __name__ == "__main__":
    main()
