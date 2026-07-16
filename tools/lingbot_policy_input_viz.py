from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def valid_depth(depth: np.ndarray, max_depth: float) -> np.ndarray:
    return np.isfinite(depth) & (depth > 0.0) & (depth < max_depth)


def colorize_depth(depth: np.ndarray, lo: float, hi: float, invalid_black: bool = True) -> np.ndarray:
    depth = depth.astype(np.float32, copy=False)
    valid = np.isfinite(depth) & (depth > 0.0)
    clean = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.clip((clean - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    if invalid_black:
        color[~valid] = 0
    return color


def resize_model_input(rgb: np.ndarray, depth: np.ndarray, max_side: int) -> tuple[np.ndarray, np.ndarray, float]:
    h, w = depth.shape
    if max_side <= 0 or max(h, w) <= max_side:
        return rgb, depth, 1.0
    scale = max_side / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    rgb_small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    depth_small = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return rgb_small, depth_small, scale


def normalized_intrinsics(width: int, height: int, scale: float, device: torch.device) -> torch.Tensor:
    fx = 690.1201171875 * scale
    fy = 689.968994140625 * scale
    cx = 641.6088256835938 * scale
    cy = 358.8370056152344 * scale
    k = np.array(
        [[fx / width, 0.0, cx / width], [0.0, fy / height, cy / height], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return torch.tensor(k, dtype=torch.float32, device=device)[None]


def prepare_tensors(rgb: np.ndarray, depth: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    image = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)[None]
    depth_clean = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    depth_clean[depth_clean < 0.0] = 0.0
    depth_t = torch.tensor(depth_clean, dtype=torch.float32, device=device)[None]
    return image, depth_t


def load_model(repo_path: Path, model_name: str, device: torch.device):
    sys.path.insert(0, str(repo_path))
    from mdm.model.v2 import MDMModel

    model = MDMModel.from_pretrained(model_name).to(device)
    if hasattr(model, "enable_pytorch_native_sdpa"):
        model.enable_pytorch_native_sdpa()
    model.eval()
    return model


def infer_lingbot(
    model,
    rgb: np.ndarray,
    depth: np.ndarray,
    max_side: int,
    num_tokens: int,
    resolution_level: int,
    device: torch.device,
    use_fp16: bool,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    rgb_in, depth_in, scale = resize_model_input(rgb, depth, max_side)
    h, w = depth_in.shape
    image_t, depth_t = prepare_tensors(rgb_in, depth_in, device)
    intrinsics_t = normalized_intrinsics(w, h, scale, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.infer(
            image_t,
            depth_in=depth_t,
            intrinsics=intrinsics_t,
            num_tokens=num_tokens if num_tokens > 0 else None,
            resolution_level=resolution_level,
            apply_mask=False,
            use_fp16=use_fp16,
            enable_depth_mask=False,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    pred = output["depth"].squeeze(0).detach().float().cpu().numpy().astype(np.float32)
    return pred, elapsed_ms, (w, h)


def preprocess_policy_input(
    depth_m: np.ndarray,
    width: int,
    height: int,
    crop_up: int,
    crop_down: int,
    crop_left: int,
    crop_right: int,
    depth_min: float,
    depth_max: float,
    inpaint_threshold: float,
    inpaint_radius: float,
    gaussian_kernel: int,
    gaussian_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    downsampled = cv2.resize(depth_m, (width, height), interpolation=cv2.INTER_NEAREST)
    bottom = height - crop_down
    right = width - crop_right
    crop = np.ascontiguousarray(downsampled[crop_up:bottom, crop_left:right], dtype=np.float32)

    processed_m = crop.copy()
    invalid = ~np.isfinite(processed_m) | (processed_m < inpaint_threshold)
    processed_m[~np.isfinite(processed_m)] = 0.0
    if inpaint_radius > 0.0 and invalid.any():
        processed_m = cv2.inpaint(processed_m, invalid.astype(np.uint8), float(inpaint_radius), cv2.INPAINT_NS)
    if gaussian_kernel > 0:
        if gaussian_kernel % 2 == 0:
            gaussian_kernel += 1
        processed_m = cv2.GaussianBlur(processed_m, (gaussian_kernel, gaussian_kernel), gaussian_sigma)

    processed_m = np.nan_to_num(processed_m, nan=depth_max, posinf=depth_max, neginf=depth_min)
    processed = np.clip(processed_m, depth_min, depth_max)
    processed = (processed - depth_min) / max(depth_max - depth_min, 1e-6)
    return downsampled, crop, np.ascontiguousarray(processed, dtype=np.float32)


def fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    panel[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return panel


def put_label(panel: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    out = panel.copy()
    bar_h = 38 if subtitle else 24
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(out, subtitle, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def depth_stats_text(raw64: np.ndarray, pred64: np.ndarray, max_depth: float) -> str:
    mask = valid_depth(raw64, max_depth) & valid_depth(pred64, max_depth)
    if not mask.any():
        return "MAE n/a"
    err = np.abs(raw64[mask] - pred64[mask])
    return f"MAE {np.mean(err) * 100:.1f}cm P90 {np.percentile(err, 90) * 100:.1f}cm"


def make_canvas(
    rgb: np.ndarray,
    raw_depth: np.ndarray,
    lingbot_depth: np.ndarray,
    infer_ms: float,
    model_size: tuple[int, int],
    args: argparse.Namespace,
) -> np.ndarray:
    raw64, raw_crop, raw_policy = preprocess_policy_input(
        raw_depth,
        args.policy_width,
        args.policy_height,
        args.crop_up,
        args.crop_down,
        args.crop_left,
        args.crop_right,
        args.depth_min,
        args.depth_max,
        args.inpaint_threshold,
        args.inpaint_radius,
        args.gaussian_kernel,
        args.gaussian_sigma,
    )
    ling64, ling_crop, ling_policy = preprocess_policy_input(
        lingbot_depth,
        args.policy_width,
        args.policy_height,
        args.crop_up,
        args.crop_down,
        args.crop_left,
        args.crop_right,
        args.depth_min,
        args.depth_max,
        args.inpaint_threshold,
        args.inpaint_radius,
        args.gaussian_kernel,
        args.gaussian_sigma,
    )
    crop_shape = f"{raw_crop.shape[1]}x{raw_crop.shape[0]}"
    row0 = [
        put_label(fit_panel(rgb, args.panel_width, args.panel_height), "RAW baseline", "RGB"),
        put_label(fit_panel(colorize_depth(raw_depth, 0.1, args.depth_vis_max), args.panel_width, args.panel_height), "Input depth", "raw native"),
        put_label(fit_panel(colorize_depth(raw64, 0.1, args.depth_vis_max), args.panel_width, args.panel_height), "Downsample", f"{args.policy_width}x{args.policy_height}"),
        put_label(fit_panel(colorize_depth(raw_crop, 0.1, args.depth_vis_max), args.panel_width, args.panel_height), "Crop", f"{crop_shape} meters"),
        put_label(fit_panel(colorize_depth(raw_policy, 0.0, 1.0, invalid_black=False), args.panel_width, args.panel_height), "MC policy input", "raw baseline"),
    ]
    row1 = [
        put_label(fit_panel(rgb, args.panel_width, args.panel_height), "LINGBOT", f"max_side {args.max_side}, tokens {args.num_tokens}"),
        put_label(
            fit_panel(colorize_depth(lingbot_depth, 0.1, args.depth_vis_max), args.panel_width, args.panel_height),
            "LingBot depth",
            f"{model_size[0]}x{model_size[1]} | {infer_ms:.1f} ms",
        ),
        put_label(
            fit_panel(colorize_depth(ling64, 0.1, args.depth_vis_max), args.panel_width, args.panel_height),
            "Downsample",
            depth_stats_text(raw64, ling64, args.max_error_depth),
        ),
        put_label(fit_panel(colorize_depth(ling_crop, 0.1, args.depth_vis_max), args.panel_width, args.panel_height), "Crop", f"{crop_shape} meters"),
        put_label(
            fit_panel(colorize_depth(ling_policy, 0.0, 1.0, invalid_black=False), args.panel_width, args.panel_height),
            "MC policy input",
            "LingBot final",
        ),
    ]
    canvas = np.concatenate([np.concatenate(row0, axis=1), np.concatenate(row1, axis=1)], axis=0)
    note = "no calibration | no inpaint | no gaussian"
    cv2.putText(canvas, note, (8, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def make_ffmpeg_writer(path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def close_writer(proc: subprocess.Popen) -> None:
    assert proc.stdin is not None
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed with code {rc}")


def process_one(model, input_path: Path, output_dir: Path, stem: str, args: argparse.Namespace, device: torch.device) -> dict:
    with np.load(input_path, allow_pickle=False) as data:
        rgbs = data["rgb"]
        depths = data["depth"].astype(np.float32)
    if args.limit > 0:
        rgbs = rgbs[: args.limit]
        depths = depths[: args.limit]

    warmup_n = min(args.warmup, len(rgbs))
    model_sizes: list[tuple[int, int]] = []
    for idx in range(warmup_n):
        _, ms, model_size = infer_lingbot(
            model, rgbs[idx], depths[idx], args.max_side, args.num_tokens, args.resolution_level, device, not args.no_fp16
        )
        model_sizes.append(model_size)
        print(f"warmup {stem} frame={idx} ms={ms:.2f} model_input={model_size[0]}x{model_size[1]}")

    out_path = output_dir / f"lingbot_policy_input_{stem}_maxside{args.max_side}_tokens{args.num_tokens}.mp4"
    canvas_w = args.panel_width * 5
    canvas_h = args.panel_height * 2
    writer = make_ffmpeg_writer(out_path, canvas_w, canvas_h, args.fps)

    timings: list[float] = []
    mae_cm: list[float] = []
    p90_cm: list[float] = []
    png_paths: list[str] = []
    capture_indices = {0, len(rgbs) // 2, len(rgbs) - 1}
    last_model_size = (0, 0)

    for idx, (rgb, raw) in enumerate(zip(rgbs, depths)):
        pred, infer_ms, model_size = infer_lingbot(
            model, rgb, raw, args.max_side, args.num_tokens, args.resolution_level, device, not args.no_fp16
        )
        last_model_size = model_size
        raw64 = cv2.resize(raw, (args.policy_width, args.policy_height), interpolation=cv2.INTER_NEAREST)
        pred64 = cv2.resize(pred, (args.policy_width, args.policy_height), interpolation=cv2.INTER_NEAREST)
        mask = valid_depth(raw64, args.max_error_depth) & valid_depth(pred64, args.max_error_depth)
        if mask.any():
            err = np.abs(raw64[mask] - pred64[mask])
            mae_cm.append(float(np.mean(err) * 100.0))
            p90_cm.append(float(np.percentile(err, 90) * 100.0))
        timings.append(infer_ms)
        canvas = make_canvas(rgb, raw, pred, infer_ms, model_size, args)
        assert writer.stdin is not None
        writer.stdin.write(canvas.astype(np.uint8).tobytes())
        if idx in capture_indices:
            png_path = output_dir / f"lingbot_policy_input_{stem}_frame{idx:03d}_maxside{args.max_side}_tokens{args.num_tokens}.png"
            cv2.imwrite(str(png_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            png_paths.append(str(png_path))
        if (idx + 1) % 10 == 0:
            print(f"processed {stem}: {idx + 1}/{len(rgbs)}")

    close_writer(writer)
    timings_np = np.asarray(timings, dtype=np.float64)
    summary = {
        "input": str(input_path),
        "output": str(out_path),
        "frames": int(len(rgbs)),
        "fps": args.fps,
        "model": args.model,
        "max_side": args.max_side,
        "num_tokens": args.num_tokens,
        "resolution_level": args.resolution_level,
        "model_input_width": int(last_model_size[0]),
        "model_input_height": int(last_model_size[1]),
        "policy_width": args.policy_width,
        "policy_height": args.policy_height,
        "crop_up": args.crop_up,
        "crop_down": args.crop_down,
        "crop_left": args.crop_left,
        "crop_right": args.crop_right,
        "crop_width": int(args.policy_width - args.crop_left - args.crop_right),
        "crop_height": int(args.policy_height - args.crop_up - args.crop_down),
        "depth_min": args.depth_min,
        "depth_max": args.depth_max,
        "inpaint_radius": args.inpaint_radius,
        "gaussian_kernel": args.gaussian_kernel,
        "timing_ms_mean": float(timings_np.mean()),
        "timing_ms_p50": float(np.percentile(timings_np, 50)),
        "timing_ms_p95": float(np.percentile(timings_np, 95)),
        "timing_ms_max": float(timings_np.max()),
        "estimated_processing_fps": float(1000.0 / timings_np.mean()),
        "mae_64x36_cm_mean": float(np.mean(mae_cm)) if mae_cm else None,
        "p90_64x36_cm_mean": float(np.mean(p90_cm)) if p90_cm else None,
        "cuda_peak_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else None,
        "png_paths": png_paths,
    }
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.home() / "lingbot-depth")
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--stem", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="robbyant/lingbot-depth-postrain-dc-vitl14")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-side", type=int, default=320)
    parser.add_argument("--num-tokens", type=int, default=300)
    parser.add_argument("--resolution-level", type=int, default=3)
    parser.add_argument("--policy-width", type=int, default=64)
    parser.add_argument("--policy-height", type=int, default=36)
    parser.add_argument("--crop-up", type=int, default=18)
    parser.add_argument("--crop-down", type=int, default=0)
    parser.add_argument("--crop-left", type=int, default=16)
    parser.add_argument("--crop-right", type=int, default=16)
    parser.add_argument("--depth-min", type=float, default=0.0)
    parser.add_argument("--depth-max", type=float, default=2.5)
    parser.add_argument("--depth-vis-max", type=float, default=2.5)
    parser.add_argument("--max-error-depth", type=float, default=50.0)
    parser.add_argument("--inpaint-threshold", type=float, default=0.2)
    parser.add_argument("--inpaint-radius", type=float, default=0.0)
    parser.add_argument("--gaussian-kernel", type=int, default=0)
    parser.add_argument("--gaussian-sigma", type=float, default=0.0)
    parser.add_argument("--panel-width", type=int, default=260)
    parser.add_argument("--panel-height", type=int, default=160)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--no-fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stem and len(args.stem) != len(args.input):
        raise ValueError("--stem count must match --input count")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
        print(f"torch={torch.__version__} cuda={torch.version.cuda}")
        torch.cuda.reset_peak_memory_stats()

    load_start = time.perf_counter()
    model = load_model(args.repo, args.model, device)
    print(f"model_load_ms={(time.perf_counter() - load_start) * 1000.0:.1f}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    summaries = []
    for i, input_path in enumerate(args.input):
        stem = args.stem[i] if args.stem else input_path.parent.name.replace("input_", "")
        if stem == "input":
            stem = "115349"
        summaries.append(process_one(model, input_path, args.output_dir, stem, args, device))
    summary_path = args.output_dir / f"summary_policy_input_maxside{args.max_side}_tokens{args.num_tokens}.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
