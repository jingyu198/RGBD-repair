from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def valid_depth(depth: np.ndarray, max_depth: float) -> np.ndarray:
    return np.isfinite(depth) & (depth > 0) & (depth < max_depth)


def colorize_depth(depth: np.ndarray, valid: np.ndarray, lo: float, hi: float) -> np.ndarray:
    clean = depth.copy()
    clean[~np.isfinite(clean)] = 0
    norm = np.clip((clean - lo) / max(hi - lo, 1e-6), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color[~valid] = 0
    return color


def colorize_error(raw: np.ndarray, calibrated: np.ndarray, valid: np.ndarray, max_error_m: float) -> np.ndarray:
    err = np.zeros_like(raw, dtype=np.float32)
    err[valid] = np.abs(calibrated[valid] - raw[valid])
    norm = np.clip(err / max(max_error_m, 1e-6), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color[~valid] = 0
    return color


def put_label(image: np.ndarray, label: str) -> np.ndarray:
    out = image.copy()
    lines = label.split("\n")
    height = 36 + 30 * (len(lines) - 1)
    cv2.rectangle(out, (0, 0), (1280, height), (0, 0, 0), thickness=-1)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (12, 27 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def resize_inputs(rgb: np.ndarray, depth: np.ndarray, max_side: int):
    h, w = depth.shape
    if max_side <= 0 or max(h, w) <= max_side:
        return rgb, depth, 1.0
    scale = max_side / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    rgb_small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    depth_small = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return rgb_small, depth_small, scale


def normalized_intrinsics(width: int, height: int, scale: float, device: torch.device):
    fx = 690.1201171875 * scale
    fy = 689.968994140625 * scale
    cx = 641.6088256835938 * scale
    cy = 358.8370056152344 * scale
    k = np.array(
        [[fx / width, 0.0, cx / width], [0.0, fy / height, cy / height], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return torch.tensor(k, dtype=torch.float32, device=device)[None]


def prepare_tensors(rgb: np.ndarray, depth: np.ndarray, device: torch.device):
    image = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)[None]
    depth_clean = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    depth_clean[depth_clean < 0] = 0
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


def infer_one(
    model,
    rgb: np.ndarray,
    depth: np.ndarray,
    max_side: int,
    resolution_level: int,
    num_tokens: int,
    device: torch.device,
    use_fp16: bool,
):
    rgb_in, depth_in, scale = resize_inputs(rgb, depth, max_side)
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
    pred = output["depth"].squeeze(0).detach().float().cpu().numpy()
    if pred.shape != depth.shape:
        pred = cv2.resize(pred, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
    return pred.astype(np.float32), elapsed_ms


def reliable_raw_mask(raw: np.ndarray, pred: np.ndarray, max_depth: float) -> np.ndarray:
    raw_valid = valid_depth(raw, max_depth)
    pred_valid = valid_depth(pred, max_depth)
    # Avoid using object boundaries and sensor edge artifacts for metric calibration.
    eroded = cv2.erode(raw_valid.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    raw_med = cv2.medianBlur(raw.astype(np.float32), 5)
    discontinuity = np.abs(raw - raw_med) > np.maximum(0.08, raw * 0.04)
    return eroded & pred_valid & ~discontinuity


def robust_scale_bias(pred: np.ndarray, raw: np.ndarray, mask: np.ndarray, max_samples: int, rng: np.random.Generator):
    ys, xs = np.nonzero(mask)
    if ys.size < 200:
        return 1.0, 0.0, int(ys.size), np.nan, np.nan

    if ys.size > max_samples:
        idx = rng.choice(ys.size, max_samples, replace=False)
        ys, xs = ys[idx], xs[idx]

    x = pred[ys, xs].astype(np.float64)
    y = raw[ys, xs].astype(np.float64)
    good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[good], y[good]
    if x.size < 200:
        return 1.0, 0.0, int(x.size), np.nan, np.nan

    # First pass affine fit.
    a, b = np.polyfit(x, y, 1)
    residual = y - (a * x + b)
    med = np.median(residual)
    mad = np.median(np.abs(residual - med)) + 1e-6
    inliers = np.abs(residual - med) < 3.0 * 1.4826 * mad

    if int(inliers.sum()) >= 200:
        a, b = np.polyfit(x[inliers], y[inliers], 1)
        residual = y[inliers] - (a * x[inliers] + b)
        inlier_count = int(inliers.sum())
    else:
        inlier_count = int(x.size)

    # Keep the learned prior close to sensor scale; extreme affine fits are almost always bad calibration.
    a = float(np.clip(a, 0.65, 1.45))
    b = float(np.clip(b, -1.0, 1.0))
    med_abs_err = float(np.median(np.abs(residual))) if residual.size else np.nan
    p90_abs_err = float(np.percentile(np.abs(residual), 90)) if residual.size else np.nan
    return a, b, inlier_count, med_abs_err, p90_abs_err


def calibrate_depth(
    raw: np.ndarray,
    pred: np.ndarray,
    max_depth: float,
    rng: np.random.Generator,
    max_samples: int,
    calib_downscale: int,
    eval_downscale: int,
):
    raw_valid = valid_depth(raw, max_depth)
    pred_valid = valid_depth(pred, max_depth)

    if calib_downscale > 1:
        h, w = raw.shape
        small_size = (max(1, w // calib_downscale), max(1, h // calib_downscale))
        raw_calib = cv2.resize(raw, small_size, interpolation=cv2.INTER_NEAREST)
        pred_calib = cv2.resize(pred, small_size, interpolation=cv2.INTER_NEAREST)
    else:
        raw_calib = raw
        pred_calib = pred

    calib_mask = reliable_raw_mask(raw_calib, pred_calib, max_depth)
    scale, bias, n_inliers, med_err, p90_err = robust_scale_bias(
        pred_calib, raw_calib, calib_mask, max_samples, rng
    )
    calibrated = (scale * pred + bias).astype(np.float32)
    calibrated_valid = valid_depth(calibrated, max_depth)

    if eval_downscale > 1:
        raw_eval = raw[::eval_downscale, ::eval_downscale]
        calibrated_eval = calibrated[::eval_downscale, ::eval_downscale]
    else:
        raw_eval = raw
        calibrated_eval = calibrated
    eval_mask = valid_depth(raw_eval, max_depth) & valid_depth(calibrated_eval, max_depth)
    if eval_mask.any():
        err = np.abs(calibrated_eval[eval_mask] - raw_eval[eval_mask])
        mae = float(np.mean(err))
        medae = float(np.median(err))
        p90 = float(np.percentile(err, 90))
        rmse = float(np.sqrt(np.mean(np.square(err))))
    else:
        mae = medae = p90 = rmse = np.nan

    return calibrated, {
        "scale": scale,
        "bias": bias,
        "calib_pixels": int(calib_mask.sum()),
        "calib_inliers": n_inliers,
        "calib_med_abs_err_m": med_err,
        "calib_p90_abs_err_m": p90_err,
        "eval_mae_m": mae,
        "eval_medae_m": medae,
        "eval_p90_m": p90,
        "eval_rmse_m": rmse,
        "eval_pixels": int(eval_mask.sum()),
        "raw_valid_pct": float(raw_valid.mean() * 100.0),
        "pred_valid_pct": float(pred_valid.mean() * 100.0),
        "calibrated_valid_pct": float(calibrated_valid.mean() * 100.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.home() / "lingbot-depth")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="robbyant/lingbot-depth-postrain-dc-vitl14")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--max-side", type=int, default=320)
    parser.add_argument("--resolution-level", type=int, default=3)
    parser.add_argument("--num-tokens", type=int, default=600)
    parser.add_argument("--preview-scale", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--calib-downscale", type=int, default=8)
    parser.add_argument("--eval-downscale", type=int, default=8)
    parser.add_argument("--error-max-cm", type=float, default=20.0)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
        print(f"torch={torch.__version__} cuda={torch.version.cuda}")
        torch.cuda.reset_peak_memory_stats()

    with np.load(args.input, allow_pickle=False) as data:
        rgbs = data["rgb"]
        depths = data["depth"].astype(np.float32)
    if args.limit > 0:
        rgbs = rgbs[: args.limit]
        depths = depths[: args.limit]

    model_load_start = time.perf_counter()
    model = load_model(args.repo, args.model, device)
    model_load_ms = (time.perf_counter() - model_load_start) * 1000.0
    print(f"model_load_ms={model_load_ms:.1f}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for idx in range(min(args.warmup, len(rgbs))):
        _, ms = infer_one(model, rgbs[idx], depths[idx], args.max_side, args.resolution_level, args.num_tokens, device, not args.no_fp16)
        print(f"warmup frame={idx} ms={ms:.1f}")

    valid_all = valid_depth(depths, args.max_depth)
    vals = depths[valid_all]
    lo, hi = np.percentile(vals, [2, 98])

    h, w = depths.shape[1:]
    out_w = int(w * 4 * args.preview_scale)
    out_h = int(h * args.preview_scale)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {args.output}")

    rng = np.random.default_rng(20260602)
    timings = []
    fusion_timings = []
    stats = []

    for idx, (rgb, raw) in enumerate(zip(rgbs, depths)):
        pred, infer_ms = infer_one(model, rgb, raw, args.max_side, args.resolution_level, args.num_tokens, device, not args.no_fp16)
        t0 = time.perf_counter()
        calibrated, frame_stats = calibrate_depth(
            raw, pred, args.max_depth, rng, args.max_samples, args.calib_downscale, args.eval_downscale
        )
        fusion_ms = (time.perf_counter() - t0) * 1000.0
        frame_ms = infer_ms + fusion_ms
        timings.append(frame_ms)
        fusion_timings.append(fusion_ms)
        frame_stats["frame_ms"] = frame_ms
        frame_stats["frame_fps"] = 1000.0 / frame_ms if frame_ms > 0 else np.nan
        stats.append(frame_stats)

        raw_valid = valid_depth(raw, args.max_depth)
        calibrated_valid = valid_depth(calibrated, args.max_depth)
        eval_valid = raw_valid & calibrated_valid
        panels = [
            put_label(rgb, f"RGB | frame {idx:03d}"),
            put_label(colorize_depth(raw, raw_valid, lo, hi), f"Raw metric | valid {raw_valid.mean()*100:.1f}%"),
            put_label(
                colorize_depth(calibrated, calibrated_valid, lo, hi),
                "LingBot calibrated | "
                f"FPS {frame_stats['frame_fps']:.1f} | a={frame_stats['scale']:.3f} b={frame_stats['bias']:.2f}\n"
                f"Err vs raw-valid GT: MAE {frame_stats['eval_mae_m']*100:.1f}cm "
                f"Med {frame_stats['eval_medae_m']*100:.1f}cm P90 {frame_stats['eval_p90_m']*100:.1f}cm",
            ),
            put_label(
                colorize_error(raw, calibrated, eval_valid, args.error_max_cm / 100.0),
                f"Error heatmap | abs(calibrated - raw GT)\n"
                f"blue=low red=high | range 0-{args.error_max_cm:.0f}cm | black=no GT",
            ),
        ]
        frame = np.concatenate(panels, axis=1)
        if args.preview_scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if (idx + 1) % 10 == 0:
            print(f"processed {idx + 1}/{len(rgbs)} frames")

    writer.release()

    timings_np = np.array(timings)
    fusion_np = np.array(fusion_timings)
    summary = {
        "model": args.model,
        "frames": len(rgbs),
        "fps": args.fps,
        "output": str(args.output),
        "max_side": args.max_side,
        "num_tokens": args.num_tokens,
        "max_depth": args.max_depth,
        "calib_downscale": args.calib_downscale,
        "eval_downscale": args.eval_downscale,
        "error_max_cm": args.error_max_cm,
        "model_load_ms": model_load_ms,
        "timing_ms_mean": float(timings_np.mean()),
        "timing_ms_p50": float(np.percentile(timings_np, 50)),
        "timing_ms_p95": float(np.percentile(timings_np, 95)),
        "timing_ms_max": float(timings_np.max()),
        "fusion_ms_mean": float(fusion_np.mean()),
        "estimated_processing_fps": float(1000.0 / timings_np.mean()),
        "scale_mean": float(np.mean([s["scale"] for s in stats])),
        "scale_p05_p95": [float(x) for x in np.percentile([s["scale"] for s in stats], [5, 95])],
        "bias_mean_m": float(np.mean([s["bias"] for s in stats])),
        "calib_med_abs_err_m_mean": float(np.nanmean([s["calib_med_abs_err_m"] for s in stats])),
        "calib_p90_abs_err_m_mean": float(np.nanmean([s["calib_p90_abs_err_m"] for s in stats])),
        "eval_mae_m_mean": float(np.nanmean([s["eval_mae_m"] for s in stats])),
        "eval_medae_m_mean": float(np.nanmean([s["eval_medae_m"] for s in stats])),
        "eval_p90_m_mean": float(np.nanmean([s["eval_p90_m"] for s in stats])),
        "eval_rmse_m_mean": float(np.nanmean([s["eval_rmse_m"] for s in stats])),
        "raw_valid_pct_mean": float(np.mean([s["raw_valid_pct"] for s in stats])),
        "calibrated_valid_pct_mean": float(np.mean([s["calibrated_valid_pct"] for s in stats])),
        "cuda_peak_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else None,
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
