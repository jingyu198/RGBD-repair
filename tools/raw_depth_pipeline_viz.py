from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PYDEPS = os.environ.get("DEPTH_VIZ_PYDEPS")
if PYDEPS and Path(PYDEPS).exists():
    sys.path.insert(0, PYDEPS)

import cv2
import numpy as np


def depth_to_gray(depth: np.ndarray, lo: float, hi: float, invalid_black: bool = True) -> np.ndarray:
    depth = depth.astype(np.float32, copy=False)
    valid = np.isfinite(depth) & (depth > 0.0)
    clean = np.nan_to_num(depth, nan=0.0, posinf=hi, neginf=lo)
    gray = np.clip((clean - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    gray_u8 = (gray * 255.0).astype(np.uint8)
    if invalid_black:
        gray_u8[~valid] = 0
    return np.repeat(gray_u8[..., None], 3, axis=2)


def mask_to_panel(mask: np.ndarray) -> np.ndarray:
    gray = (mask.astype(np.uint8) * 255)
    return np.repeat(gray[..., None], 3, axis=2)


def fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    panel[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return panel


def label(panel: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    out = panel.copy()
    bar_h = 40 if subtitle else 25
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(out, subtitle, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def process_depth(depth: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    down = cv2.resize(depth.astype(np.float32), (args.width, args.height), interpolation=cv2.INTER_NEAREST)
    bottom = args.height - args.crop_down
    right = args.width - args.crop_right
    crop = np.ascontiguousarray(down[args.crop_up:bottom, args.crop_left:right], dtype=np.float32)

    pre_inpaint = crop.copy()
    invalid = ~np.isfinite(pre_inpaint) | (pre_inpaint < args.inpaint_threshold)
    pre_inpaint[~np.isfinite(pre_inpaint)] = 0.0
    pre_inpaint[invalid] = 0.0

    if args.inpaint_radius > 0 and invalid.any():
        inpainted = cv2.inpaint(pre_inpaint, invalid.astype(np.uint8), float(args.inpaint_radius), cv2.INPAINT_NS)
    else:
        inpainted = pre_inpaint.copy()

    blurred = inpainted.copy()
    if args.gaussian_kernel > 0:
        k = args.gaussian_kernel if args.gaussian_kernel % 2 == 1 else args.gaussian_kernel + 1
        blurred = cv2.GaussianBlur(blurred, (k, k), args.gaussian_sigma)

    cleaned = np.nan_to_num(blurred, nan=args.depth_max, posinf=args.depth_max, neginf=args.depth_min)
    clipped = np.clip(cleaned, args.depth_min, args.depth_max)
    policy = (clipped - args.depth_min) / max(args.depth_max - args.depth_min, 1e-6)
    return {
        "down": down,
        "crop": crop,
        "invalid": invalid,
        "inpainted": inpainted.astype(np.float32),
        "blurred": blurred.astype(np.float32),
        "policy": policy.astype(np.float32),
    }


def make_canvas(depth: np.ndarray, frame_idx: int, stem: str, args: argparse.Namespace) -> np.ndarray:
    stages = process_depth(depth, args)
    crop_invalid_pct = float(stages["invalid"].mean() * 100.0)
    panels = [
        label(
            fit_panel(depth_to_gray(depth, args.depth_min, args.depth_vis_max), args.panel_width, args.panel_height),
            "Raw depth",
            f"native frame {frame_idx}",
        ),
        label(
            fit_panel(depth_to_gray(stages["down"], args.depth_min, args.depth_vis_max), args.panel_width, args.panel_height),
            "Downsample",
            f"{args.width}x{args.height}",
        ),
        label(
            fit_panel(depth_to_gray(stages["crop"], args.depth_min, args.depth_vis_max), args.panel_width, args.panel_height),
            "Crop",
            f"{stages['crop'].shape[1]}x{stages['crop'].shape[0]} meters",
        ),
        label(
            fit_panel(mask_to_panel(stages["invalid"]), args.panel_width, args.panel_height),
            "Invalid mask",
            f"< {args.inpaint_threshold:.1f}m | {crop_invalid_pct:.1f}%",
        ),
        label(
            fit_panel(depth_to_gray(stages["inpainted"], args.depth_min, args.depth_vis_max), args.panel_width, args.panel_height),
            "After inpaint",
            f"radius {args.inpaint_radius:g}",
        ),
        label(
            fit_panel(depth_to_gray(stages["blurred"], args.depth_min, args.depth_vis_max), args.panel_width, args.panel_height),
            "After blur",
            f"{args.gaussian_kernel}x{args.gaussian_kernel}, sigma {args.gaussian_sigma:g}",
        ),
        label(
            fit_panel(depth_to_gray(stages["policy"], 0.0, 1.0, invalid_black=False), args.panel_width, args.panel_height),
            "Final input",
            "clip 0-2.5m -> 0-1",
        ),
    ]
    canvas = np.concatenate(panels, axis=1)
    cv2.putText(
        canvas,
        f"{stem} | raw depth pipeline | near=black far=white",
        (8, canvas.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def open_writer(path: Path, width: int, height: int, fps: float) -> tuple[cv2.VideoWriter, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if writer.isOpened():
        return writer, path
    fallback = path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(fallback), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path} or {fallback}")
    return writer, fallback


def process_demo(input_path: Path, stem: str, output_dir: Path, args: argparse.Namespace) -> dict:
    with np.load(input_path, allow_pickle=False) as data:
        depths = data["depth"].astype(np.float32)
    total = int(depths.shape[0])
    limit = total if args.limit <= 0 else min(total, args.limit)
    frames = depths[:limit]

    out_video = output_dir / f"raw_depth_pipeline_{stem}.mp4"
    canvas_w = args.panel_width * 7
    canvas_h = args.panel_height
    writer, actual_video = open_writer(out_video, canvas_w, canvas_h, args.fps)

    key_indices = sorted(set([0, limit // 2, limit - 1]))
    key_pngs: list[str] = []
    invalid_pcts: list[float] = []

    for idx, depth in enumerate(frames):
        stages = process_depth(depth, args)
        invalid_pcts.append(float(stages["invalid"].mean() * 100.0))
        canvas = make_canvas(depth, idx, stem, args)
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        if idx in key_indices:
            png_path = output_dir / f"raw_depth_pipeline_{stem}_frame{idx:03d}.png"
            cv2.imwrite(str(png_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            key_pngs.append(str(png_path))
        if (idx + 1) % 50 == 0:
            print(f"{stem}: {idx + 1}/{limit}")

    writer.release()
    summary = {
        "stem": stem,
        "input": str(input_path),
        "output_video": str(actual_video),
        "key_pngs": key_pngs,
        "frames": limit,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "crop_up": args.crop_up,
        "crop_down": args.crop_down,
        "crop_left": args.crop_left,
        "crop_right": args.crop_right,
        "crop_width": args.width - args.crop_left - args.crop_right,
        "crop_height": args.height - args.crop_up - args.crop_down,
        "inpaint_threshold_m": args.inpaint_threshold,
        "inpaint_radius": args.inpaint_radius,
        "gaussian_kernel": args.gaussian_kernel,
        "gaussian_sigma": args.gaussian_sigma,
        "depth_min": args.depth_min,
        "depth_max": args.depth_max,
        "crop_invalid_pct_mean": float(np.mean(invalid_pcts)),
        "crop_invalid_pct_p95": float(np.percentile(invalid_pcts, 95)),
    }
    (output_dir / f"raw_depth_pipeline_{stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--stem", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=36)
    parser.add_argument("--crop-up", type=int, default=18)
    parser.add_argument("--crop-down", type=int, default=0)
    parser.add_argument("--crop-left", type=int, default=16)
    parser.add_argument("--crop-right", type=int, default=16)
    parser.add_argument("--depth-min", type=float, default=0.0)
    parser.add_argument("--depth-max", type=float, default=2.5)
    parser.add_argument("--depth-vis-max", type=float, default=2.5)
    parser.add_argument("--inpaint-threshold", type=float, default=0.2)
    parser.add_argument("--inpaint-radius", type=float, default=3.0)
    parser.add_argument("--gaussian-kernel", type=int, default=3)
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument("--panel-width", type=int, default=220)
    parser.add_argument("--panel-height", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.input) != len(args.stem):
        raise ValueError("--input and --stem counts must match")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for input_path, stem in zip(args.input, args.stem):
        summaries.append(process_demo(input_path, stem, args.output_dir, args))
    summary_path = args.output_dir / "summary_raw_depth_pipeline.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
