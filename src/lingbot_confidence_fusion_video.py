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
    return np.isfinite(depth) & (depth > 0.0) & (depth < max_depth)


def resize_inputs(rgb: np.ndarray, depth: np.ndarray, max_side: int):
    h, w = depth.shape
    if max_side <= 0 or max(h, w) <= max_side:
        return rgb, depth, 1.0
    scale = max_side / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return (
        cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA),
        cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST),
        scale,
    )


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


def specular_like_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1].astype(np.float32) / 255.0
    value = hsv[..., 2].astype(np.float32) / 255.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (9, 9))
    local_std = np.sqrt(np.maximum(cv2.blur((gray - mean) ** 2, (9, 9)), 0.0)) / 255.0
    return (value > 0.84) & (saturation < 0.32) & (local_std < 0.09)


def remove_small_components(mask: np.ndarray, max_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=bool)
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= max_area:
            out[labels == label] = True
    return out


def reliable_calib_mask(rgb: np.ndarray, raw: np.ndarray, lingbot: np.ndarray, max_depth: float) -> np.ndarray:
    raw_valid = valid_depth(raw, max_depth)
    ling_valid = valid_depth(lingbot, max_depth)
    eroded = cv2.erode(raw_valid.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    raw_clean = np.nan_to_num(raw.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    raw_clean[~raw_valid] = 0.0
    raw_med = cv2.medianBlur(raw_clean, 5)
    discontinuity = np.abs(raw_clean - raw_med) > np.maximum(0.08, raw_clean * 0.04)
    dx = cv2.Sobel(raw_clean, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(raw_clean, cv2.CV_32F, 0, 1, ksize=3)
    high_grad = cv2.magnitude(dx, dy) > np.maximum(0.18, raw_clean * 0.08)
    specular = specular_like_mask(rgb)
    return eroded & ling_valid & ~discontinuity & ~high_grad & ~specular


def fit_scale_bias(pred: np.ndarray, raw: np.ndarray, mask: np.ndarray, max_samples: int, rng: np.random.Generator):
    ys, xs = np.nonzero(mask)
    if ys.size < 200:
        return 1.0, 0.0, int(ys.size), np.nan, np.nan
    if ys.size > max_samples:
        idx = rng.choice(ys.size, max_samples, replace=False)
        ys, xs = ys[idx], xs[idx]
    x = pred[ys, xs].astype(np.float64)
    y = raw[ys, xs].astype(np.float64)
    good = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    x, y = x[good], y[good]
    if x.size < 200:
        return 1.0, 0.0, int(x.size), np.nan, np.nan

    a, b = np.polyfit(x, y, 1)
    residual = y - (a * x + b)
    for _ in range(2):
        med = np.median(residual)
        mad = np.median(np.abs(residual - med)) + 1e-6
        inliers = np.abs(residual - med) < 3.0 * 1.4826 * mad
        if int(inliers.sum()) < 200:
            break
        x, y = x[inliers], y[inliers]
        a, b = np.polyfit(x, y, 1)
        residual = y - (a * x + b)

    a = float(np.clip(a, 0.65, 1.45))
    b = float(np.clip(b, -1.0, 1.0))
    residual = y - (a * x + b)
    med_abs_err = float(np.median(np.abs(residual))) if residual.size else np.nan
    p90_abs_err = float(np.percentile(np.abs(residual), 90)) if residual.size else np.nan
    return a, b, int(x.size), med_abs_err, p90_abs_err


def calibrate_lingbot(
    rgb: np.ndarray,
    raw: np.ndarray,
    lingbot: np.ndarray,
    max_depth: float,
    max_samples: int,
    calib_downscale: int,
    rng: np.random.Generator,
):
    if calib_downscale > 1:
        h, w = raw.shape
        small_size = (max(1, w // calib_downscale), max(1, h // calib_downscale))
        rgb_c = cv2.resize(rgb, small_size, interpolation=cv2.INTER_AREA)
        raw_c = cv2.resize(raw, small_size, interpolation=cv2.INTER_NEAREST)
        ling_c = cv2.resize(lingbot, small_size, interpolation=cv2.INTER_NEAREST)
    else:
        rgb_c, raw_c, ling_c = rgb, raw, lingbot
    mask = reliable_calib_mask(rgb_c, raw_c, ling_c, max_depth)
    scale, bias, inliers, medae, p90 = fit_scale_bias(ling_c, raw_c, mask, max_samples, rng)
    calibrated = (scale * lingbot + bias).astype(np.float32)
    calibrated[~np.isfinite(calibrated)] = 0.0
    calibrated[(calibrated <= 0.0) | (calibrated >= max_depth)] = 0.0
    return calibrated, {
        "scale": scale,
        "bias_m": bias,
        "calib_pixels": int(mask.sum()),
        "calib_inliers": int(inliers),
        "calib_fit_medae_m": medae,
        "calib_fit_p90_m": p90,
    }


def confidence_fusion(
    rgb: np.ndarray,
    raw: np.ndarray,
    lingbot: np.ndarray,
    max_depth: float,
    diff_abs: float,
    diff_rel: float,
    noise_abs: float,
    noise_rel: float,
    small_area: int,
    calibrate: bool,
    aggressive: bool,
    invalid_dilate: int,
    specular_dilate: int,
    max_samples: int,
    calib_downscale: int,
    rng: np.random.Generator,
):
    raw_valid = valid_depth(raw, max_depth)
    if calibrate:
        lingbot_used, cal_stats = calibrate_lingbot(
            rgb, raw, lingbot, max_depth, max_samples, calib_downscale, rng
        )
    else:
        lingbot_used = lingbot
        cal_stats = {
            "scale": 1.0,
            "bias_m": 0.0,
            "calib_pixels": 0,
            "calib_inliers": 0,
            "calib_fit_medae_m": np.nan,
            "calib_fit_p90_m": np.nan,
        }
    ling_valid = valid_depth(lingbot_used, max_depth)
    raw_clean = np.nan_to_num(raw.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    raw_clean[~raw_valid] = 0.0

    raw_med = cv2.medianBlur(raw_clean, 5)
    residual = np.abs(raw_clean - raw_med)
    anomaly = raw_valid & (residual > np.maximum(noise_abs, raw_clean * noise_rel))
    small_noise = remove_small_components(anomaly, small_area)

    # Depth-camera reflective floors often produce valid-but-wrong pits. Use RGB specular cues and
    # large disagreement with LingBot as a conservative replacement trigger.
    specular = specular_like_mask(rgb)
    if specular_dilate > 0:
        k = specular_dilate if specular_dilate % 2 == 1 else specular_dilate + 1
        specular = cv2.dilate(specular.astype(np.uint8), np.ones((k, k), np.uint8), iterations=1).astype(bool)

    diff = np.zeros_like(raw_clean, dtype=np.float32)
    both_valid = raw_valid & ling_valid
    diff[both_valid] = np.abs(raw_clean[both_valid] - lingbot_used[both_valid])
    diff_large = both_valid & (diff > np.maximum(diff_abs, raw_clean * diff_rel))

    invalid_core = ~raw_valid
    if invalid_dilate > 0:
        k = invalid_dilate if invalid_dilate % 2 == 1 else invalid_dilate + 1
        invalid_zone = cv2.dilate(invalid_core.astype(np.uint8), np.ones((k, k), np.uint8), iterations=1).astype(bool)
    else:
        invalid_zone = invalid_core

    invalid_replace = invalid_zone & ling_valid
    noisy_replace = small_noise & ling_valid
    reflect_replace = specular & ling_valid & (diff_large | anomaly | invalid_zone)
    anomaly_replace = anomaly & diff_large & ling_valid
    diff_replace = diff_large & ling_valid if aggressive else np.zeros_like(diff_large, dtype=bool)
    if aggressive:
        noisy_replace = (noisy_replace | (anomaly & ling_valid))
    replace = invalid_replace | noisy_replace | reflect_replace | anomaly_replace | diff_replace

    fused = raw_clean.copy()
    fused[replace] = lingbot_used[replace]
    fused[~np.isfinite(fused)] = 0.0
    fused[(fused < 0.0) | (fused >= max_depth)] = 0.0

    masks = {
        "raw_valid": raw_valid,
        "ling_valid": ling_valid,
        "replace": replace,
        "invalid_replace": invalid_replace,
        "noisy_replace": noisy_replace,
        "reflect_replace": reflect_replace,
        "anomaly_replace": anomaly_replace,
        "diff_replace": diff_replace,
        "specular": specular,
        "diff_large": diff_large,
        "diff": diff,
        "lingbot_used": lingbot_used,
    }
    return fused.astype(np.float32), masks, cal_stats


def colorize_depth_gray(depth: np.ndarray, valid: np.ndarray, lo: float, hi: float) -> np.ndarray:
    clean = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.clip((clean - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    out[~valid] = 0
    return out


def colorize_change(raw: np.ndarray, fused: np.ndarray, mask: np.ndarray, max_depth: float, max_diff_m: float) -> np.ndarray:
    valid = valid_depth(fused, max_depth)
    diff = np.zeros_like(fused, dtype=np.float32)
    raw_valid = valid_depth(raw, max_depth)
    common = raw_valid & valid
    diff[common] = np.abs(fused[common] - raw[common])
    norm = np.clip(diff / max(max_diff_m, 1e-6), 0.0, 1.0)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color[~common] = 0
    color[mask] = (255, 0, 0)
    return color


def colorize_mask(masks: dict[str, np.ndarray]) -> np.ndarray:
    h, w = masks["replace"].shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[masks["raw_valid"]] = (40, 120, 40)
    out[masks["invalid_replace"]] = (0, 100, 255)
    out[masks["noisy_replace"]] = (255, 220, 0)
    out[masks["reflect_replace"]] = (255, 0, 0)
    out[masks["anomaly_replace"] & ~masks["reflect_replace"]] = (255, 80, 180)
    out[masks["diff_replace"] & ~masks["reflect_replace"] & ~masks["anomaly_replace"]] = (180, 80, 255)
    return out


def fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    out = np.zeros((height, width, 3), dtype=np.uint8)
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    out[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return out


def put_label(image: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 50), (0, 0, 0), thickness=-1)
    cv2.putText(out, title, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(out, subtitle, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (225, 225, 225), 1, cv2.LINE_AA)
    return out


def make_frame(
    rgb: np.ndarray,
    raw: np.ndarray,
    lingbot: np.ndarray,
    fused: np.ndarray,
    masks: dict[str, np.ndarray],
    cal_stats: dict,
    frame_idx: int,
    lo: float,
    hi: float,
    max_depth: float,
    max_diff_m: float,
    panel_w: int,
    panel_h: int,
) -> np.ndarray:
    replace_pct = masks["replace"].mean() * 100.0
    reflect_pct = masks["reflect_replace"].mean() * 100.0
    noisy_pct = masks["noisy_replace"].mean() * 100.0
    diff_pct = masks["diff_replace"].mean() * 100.0
    lingbot_used = masks.get("lingbot_used", lingbot)
    panels = [
        put_label(fit_panel(rgb, panel_w, panel_h), "RGB", f"frame {frame_idx:03d}"),
        put_label(
            fit_panel(colorize_depth_gray(raw, masks["raw_valid"], lo, hi), panel_w, panel_h),
            "Raw depth",
            "kept where reliable",
        ),
        put_label(
            fit_panel(colorize_depth_gray(lingbot_used, masks["ling_valid"], lo, hi), panel_w, panel_h),
            "Calibrated LingBot",
            f"a={cal_stats['scale']:.3f} b={cal_stats['bias_m']*100:.1f}cm",
        ),
        put_label(
            fit_panel(colorize_mask(masks), panel_w, panel_h),
            "Fusion mask",
            "green keep raw, blue invalid, purple diff, red reflect",
        ),
        put_label(
            fit_panel(colorize_depth_gray(fused, valid_depth(fused, max_depth), lo, hi), panel_w, panel_h),
            "Fused depth",
            f"replace {replace_pct:.1f}% diff {diff_pct:.1f}% reflect {reflect_pct:.1f}% noise {noisy_pct:.1f}%",
        ),
        put_label(
            fit_panel(colorize_change(raw, fused, masks["replace"], max_depth, max_diff_m), panel_w, panel_h),
            "Change map",
            f"red overlay=used LingBot, heat=|fused-raw| 0-{max_diff_m*100:.0f}cm",
        ),
    ]
    return np.concatenate([np.concatenate(panels[:3], axis=1), np.concatenate(panels[3:], axis=1)], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.home() / "lingbot-depth")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="")
    parser.add_argument("--model", default="robbyant/lingbot-depth-postrain-dc-vitl14")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--max-side", type=int, default=320)
    parser.add_argument("--resolution-level", type=int, default=3)
    parser.add_argument("--num-tokens", type=int, default=300)
    parser.add_argument("--limit", type=int, default=90)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--diff-abs", type=float, default=0.08)
    parser.add_argument("--diff-rel", type=float, default=0.04)
    parser.add_argument("--noise-abs", type=float, default=0.06)
    parser.add_argument("--noise-rel", type=float, default=0.035)
    parser.add_argument("--small-area", type=int, default=2000)
    parser.add_argument("--calibrate-lingbot", action="store_true")
    parser.add_argument("--aggressive", action="store_true")
    parser.add_argument("--invalid-dilate", type=int, default=3)
    parser.add_argument("--specular-dilate", type=int, default=11)
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--calib-downscale", type=int, default=4)
    parser.add_argument("--change-max-cm", type=float, default=25.0)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--panel-height", type=int, default=270)
    parser.add_argument("--save-png-count", type=int, default=4)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or args.input.parent.name
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
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

    for idx in range(min(args.warmup, len(rgbs))):
        _, ms = infer_one(model, rgbs[idx], depths[idx], args.max_side, args.resolution_level, args.num_tokens, device, not args.no_fp16)
        print(f"warmup frame={idx} ms={ms:.1f}")

    vals = depths[valid_depth(depths, args.max_depth)]
    lo, hi = np.percentile(vals, [2, 98]) if vals.size else (0.1, 3.0)
    lo, hi = float(max(0.0, lo)), float(min(max(hi, lo + 0.5), args.max_depth))

    mode = "confidence_fusion"
    if args.calibrate_lingbot:
        mode += "_calibrated"
    if args.aggressive:
        mode += "_aggressive"
    out_video = args.output_dir / f"lingbot_{mode}_{stem}_maxside{args.max_side}_tokens{args.num_tokens}.mp4"
    out_w = args.panel_width * 3
    out_h = args.panel_height * 2
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {out_video}")

    png_indices = set()
    if args.save_png_count > 0 and len(rgbs) > 0:
        png_indices = set(np.linspace(0, len(rgbs) - 1, min(args.save_png_count, len(rgbs)), dtype=int).tolist())

    infer_ms_values = []
    stats = []
    rng = np.random.default_rng(20260615)
    for idx, (rgb, raw) in enumerate(zip(rgbs, depths)):
        lingbot, infer_ms = infer_one(model, rgb, raw, args.max_side, args.resolution_level, args.num_tokens, device, not args.no_fp16)
        fused, masks, cal_stats = confidence_fusion(
            rgb,
            raw,
            lingbot,
            args.max_depth,
            args.diff_abs,
            args.diff_rel,
            args.noise_abs,
            args.noise_rel,
            args.small_area,
            args.calibrate_lingbot,
            args.aggressive,
            args.invalid_dilate,
            args.specular_dilate,
            args.max_samples,
            args.calib_downscale,
            rng,
        )
        infer_ms_values.append(infer_ms)
        stats.append(
            {
                "frame": idx,
                "infer_ms": infer_ms,
                "raw_valid_pct": float(masks["raw_valid"].mean() * 100.0),
                "replace_pct": float(masks["replace"].mean() * 100.0),
                "invalid_replace_pct": float(masks["invalid_replace"].mean() * 100.0),
                "noisy_replace_pct": float(masks["noisy_replace"].mean() * 100.0),
                "reflect_replace_pct": float(masks["reflect_replace"].mean() * 100.0),
                "anomaly_replace_pct": float(masks["anomaly_replace"].mean() * 100.0),
                "diff_replace_pct": float(masks["diff_replace"].mean() * 100.0),
                "fused_valid_pct": float(valid_depth(fused, args.max_depth).mean() * 100.0),
                **cal_stats,
            }
        )
        frame = make_frame(
            rgb,
            raw,
            lingbot,
            fused,
            masks,
            cal_stats,
            idx,
            lo,
            hi,
            args.max_depth,
            args.change_max_cm / 100.0,
            args.panel_width,
            args.panel_height,
        )
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if idx in png_indices:
            png_path = args.output_dir / f"lingbot_{mode}_{stem}_frame{idx:03d}.png"
            cv2.imwrite(str(png_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if (idx + 1) % 10 == 0:
            print(f"processed {idx + 1}/{len(rgbs)} frames")

    writer.release()

    summary = {
        "input": str(args.input),
        "output_video": str(out_video),
        "model": args.model,
        "frames": len(rgbs),
        "max_side": args.max_side,
        "num_tokens": args.num_tokens,
        "max_depth": args.max_depth,
        "fusion_policy": "keep raw depth by default; replace invalid/neighborhood/noisy/specular/large-disagreement regions with calibrated LingBot when enabled",
        "calibrate_lingbot": bool(args.calibrate_lingbot),
        "aggressive": bool(args.aggressive),
        "diff_abs_m": args.diff_abs,
        "diff_rel": args.diff_rel,
        "noise_abs_m": args.noise_abs,
        "noise_rel": args.noise_rel,
        "small_area": args.small_area,
        "invalid_dilate": args.invalid_dilate,
        "specular_dilate": args.specular_dilate,
        "calib_downscale": args.calib_downscale,
        "model_load_ms": model_load_ms,
        "infer_ms_mean": float(np.mean(infer_ms_values)),
        "infer_ms_p50": float(np.percentile(infer_ms_values, 50)),
        "infer_ms_p95": float(np.percentile(infer_ms_values, 95)),
        "raw_valid_pct_mean": float(np.mean([s["raw_valid_pct"] for s in stats])),
        "fused_valid_pct_mean": float(np.mean([s["fused_valid_pct"] for s in stats])),
        "replace_pct_mean": float(np.mean([s["replace_pct"] for s in stats])),
        "invalid_replace_pct_mean": float(np.mean([s["invalid_replace_pct"] for s in stats])),
        "noisy_replace_pct_mean": float(np.mean([s["noisy_replace_pct"] for s in stats])),
        "reflect_replace_pct_mean": float(np.mean([s["reflect_replace_pct"] for s in stats])),
        "anomaly_replace_pct_mean": float(np.mean([s["anomaly_replace_pct"] for s in stats])),
        "diff_replace_pct_mean": float(np.mean([s["diff_replace_pct"] for s in stats])),
        "scale_mean": float(np.mean([s["scale"] for s in stats])),
        "bias_m_mean": float(np.mean([s["bias_m"] for s in stats])),
        "calib_pixels_mean": float(np.mean([s["calib_pixels"] for s in stats])),
        "depth_visualization": {"lo_m": lo, "hi_m": hi, "mode": "gray_near_black_far_white"},
        "cuda_peak_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else None,
    }
    summary_path = args.output_dir / f"summary_{mode}_{stem}_maxside{args.max_side}_tokens{args.num_tokens}.json"
    details_path = args.output_dir / f"frames_{mode}_{stem}_maxside{args.max_side}_tokens{args.num_tokens}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    details_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
