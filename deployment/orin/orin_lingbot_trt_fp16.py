from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    cv2.rectangle(out, (0, 0), (out.shape[1], height), (0, 0, 0), thickness=-1)
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


def prepare_tensors(rgb: np.ndarray, depth: np.ndarray, device: torch.device):
    image = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)[None].contiguous()
    depth_clean = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    depth_clean[depth_clean < 0] = 0
    depth_t = torch.tensor(depth_clean, dtype=torch.float32, device=device)[None].contiguous()
    return image, depth_t


def reliable_raw_mask(raw: np.ndarray, pred: np.ndarray, max_depth: float) -> np.ndarray:
    raw_valid = valid_depth(raw, max_depth)
    pred_valid = valid_depth(pred, max_depth)
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

    a = float(np.clip(a, 0.65, 1.45))
    b = float(np.clip(b, -1.0, 1.0))
    med_abs_err = float(np.median(np.abs(residual))) if residual.size else np.nan
    p90_abs_err = float(np.percentile(np.abs(residual), 90)) if residual.size else np.nan
    return a, b, inlier_count, med_abs_err, p90_abs_err


def calibrate_depth(raw, pred, max_depth, rng, max_samples, calib_downscale, eval_downscale):
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
    scale, bias, n_inliers, med_err, p90_err = robust_scale_bias(pred_calib, raw_calib, calib_mask, max_samples, rng)
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


class LingBotDepthExport(nn.Module):
    def __init__(self, model: nn.Module, num_tokens: int):
        super().__init__()
        self.model = model
        self.num_tokens = int(num_tokens)

    def forward(self, image: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        output = self.model.forward(image, num_tokens=self.num_tokens, depth=depth, enable_depth_mask=False)
        return output["depth_reg"]


def patch_attention_for_export(model: nn.Module) -> int:
    patched = 0

    def forward_attention(self, x, attn_bias=None):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    for module in model.modules():
        if all(hasattr(module, attr) for attr in ("qkv", "num_heads", "scale", "attn_drop", "proj", "proj_drop")):
            module.forward = types.MethodType(forward_attention, module)
            patched += 1
    return patched


def patch_encoder_for_export(model: nn.Module) -> bool:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return False

    def forward_encoder(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        token_rows: int,
        token_cols: int,
        return_class_token: bool = False,
        remap_depth_in: str = "linear",
        **kwargs,
    ):
        image_14 = F.interpolate(image, (token_rows * 14, token_cols * 14), mode="bilinear", align_corners=False, antialias=False)
        image_14 = (image_14 - self.image_mean) / self.image_std

        depth_14 = F.interpolate(depth, (token_rows * 14, token_cols * 14), mode="nearest")
        dmask_14 = (depth_14 > 0.01).detach()
        depth_14 = torch.where(dmask_14, depth_14, torch.zeros_like(depth_14))

        if remap_depth_in == "linear":
            pass
        elif remap_depth_in == "log":
            depth_14 = torch.where(dmask_14, torch.log(torch.clamp(depth_14, min=1e-6)), torch.zeros_like(depth_14))
        else:
            raise NotImplementedError(f"Unsupported remap_depth_in: {remap_depth_in}")

        kwargs.pop("enable_depth_mask", None)
        features = self.backbone.get_intermediate_layers_mae(
            x_img=image_14,
            x_depth=depth_14,
            n=self.intermediate_layers,
            return_class_token=True,
            enable_depth_mask=False,
            **kwargs,
        )

        if isinstance(features[0][0], list):
            num_valid_tokens = token_rows * token_cols
            features = tuple(
                (
                    torch.cat([feat[:, :num_valid_tokens].contiguous() for feat in feats], dim=0),
                    torch.cat(cls_tokens, dim=0),
                )
                for feats, cls_tokens in features
            )

        x = torch.stack(
            [
                proj(feat.permute(0, 2, 1)[:, :, : token_rows * token_cols].unflatten(2, (token_rows, token_cols)).contiguous())
                for proj, (feat, clstoken) in zip(self.output_projections, features)
            ],
            dim=1,
        ).sum(dim=1)
        cls_token = features[-1][1]

        if return_class_token:
            return x, cls_token, None, None
        return x, None, None

    encoder.forward = types.MethodType(forward_encoder, encoder)
    if hasattr(encoder, "onnx_compatible_mode"):
        encoder.onnx_compatible_mode = True
    return True


def load_export_model(repo_path: Path, model_path: Path, device: torch.device, num_tokens: int):
    sys.path.insert(0, str(repo_path))
    from mdm.model.v2 import MDMModel

    model = MDMModel.from_pretrained(model_path).to(device)
    model.eval()
    patched_attention = patch_attention_for_export(model)
    patched_encoder = patch_encoder_for_export(model)
    wrapper = LingBotDepthExport(model, num_tokens).to(device).eval()
    return wrapper, {"patched_attention_modules": patched_attention, "patched_encoder": patched_encoder}


def export_onnx(args, input_shape):
    if args.onnx.exists() and not args.force_export:
        print(f"ONNX exists: {args.onnx}")
        return {"skipped": True}

    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    wrapper, patch_info = load_export_model(args.repo, args.model_path, device, args.num_tokens)
    batch, channels, height, width = input_shape
    image = torch.randn(batch, channels, height, width, dtype=torch.float32, device=device)
    depth = torch.rand(batch, height, width, dtype=torch.float32, device=device) + 0.1

    with torch.inference_mode():
        torch_out = wrapper(image, depth).detach().float()
    print(f"torch output shape={tuple(torch_out.shape)} min={float(torch_out.min()):.4f} max={float(torch_out.max()):.4f}")

    start = time.perf_counter()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (image, depth),
            str(args.onnx),
            input_names=["image", "depth"],
            output_names=["pred_depth"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=None,
        )
    export_ms = (time.perf_counter() - start) * 1000.0

    import onnx

    onnx_model = onnx.load(str(args.onnx))
    onnx.checker.check_model(onnx_model)
    print(f"exported ONNX: {args.onnx} ({args.onnx.stat().st_size / (1024 ** 2):.1f} MB)")
    return {"skipped": False, "export_ms": export_ms, **patch_info}


def run_trtexec(args):
    if args.engine.exists() and not args.force_engine:
        print(f"TensorRT engine exists: {args.engine}")
        return {"skipped": True}

    args.engine.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.trtexec),
        f"--onnx={args.onnx}",
        f"--saveEngine={args.engine}",
        "--fp16",
        "--builderOptimizationLevel=5",
        "--memPoolSize=workspace:4096",
        "--warmUp=500",
        "--duration=5",
        "--avgRuns=100",
    ]
    print("running:", " ".join(cmd))
    start = time.perf_counter()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    build_ms = (time.perf_counter() - start) * 1000.0
    args.trtexec_log.write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout[-8000:])
    if proc.returncode != 0:
        raise RuntimeError(f"trtexec failed with code {proc.returncode}; see {args.trtexec_log}")

    throughput = None
    gpu_compute_ms = None
    m = re.search(r"Throughput:\s*([0-9.]+)\s*qps", proc.stdout)
    if m:
        throughput = float(m.group(1))
    m = re.search(r"GPU Compute Time:\s*min\s*=\s*[0-9.]+\s*ms,\s*max\s*=\s*[0-9.]+\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms", proc.stdout)
    if m:
        gpu_compute_ms = float(m.group(1))
    return {"skipped": False, "build_ms": build_ms, "trtexec_throughput_fps": throughput, "trtexec_gpu_compute_ms": gpu_compute_ms}


def trt_dtype_to_torch(trt_dtype):
    import tensorrt as trt

    if trt_dtype == trt.float32:
        return torch.float32
    if trt_dtype == trt.float16:
        return torch.float16
    if trt_dtype == trt.int32:
        return torch.int32
    if trt_dtype == trt.int8:
        return torch.int8
    if hasattr(trt, "bool") and trt_dtype == trt.bool:
        return torch.bool
    raise TypeError(f"Unsupported TensorRT dtype: {trt_dtype}")


class TrtRunner:
    def __init__(self, engine_path: Path, device: torch.device):
        import tensorrt as trt

        self.trt = trt
        self.device = device
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.input_names = []
        self.output_names = []
        self.tensor_info = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            self.tensor_info[name] = {"shape": shape, "dtype": str(dtype), "mode": str(mode)}
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        if len(self.input_names) != 2 or len(self.output_names) != 1:
            raise RuntimeError(f"Unexpected TensorRT IO: inputs={self.input_names} outputs={self.output_names}")

    def infer(self, image: torch.Tensor, depth: torch.Tensor):
        inputs = {"image": image, "depth": depth}
        for name in self.input_names:
            tensor = inputs[name]
            if any(dim < 0 for dim in self.engine.get_tensor_shape(name)):
                self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        out_name = self.output_names[0]
        out_shape = tuple(self.context.get_tensor_shape(out_name))
        if any(dim < 0 for dim in out_shape):
            out_shape = tuple(self.engine.get_tensor_shape(out_name))
        out_dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(out_name))
        output = torch.empty(out_shape, dtype=out_dtype, device=self.device).contiguous()
        self.context.set_tensor_address(out_name, int(output.data_ptr()))

        torch.cuda.synchronize()
        start = time.perf_counter()
        ok = self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return output.float().detach().cpu().numpy(), elapsed_ms


def infer_one_trt(runner: TrtRunner, rgb: np.ndarray, depth: np.ndarray, max_side: int, device: torch.device):
    rgb_in, depth_in, _ = resize_inputs(rgb, depth, max_side)
    image_t, depth_t = prepare_tensors(rgb_in, depth_in, device)
    pred, infer_ms = runner.infer(image_t, depth_t)
    pred = pred.squeeze(0)
    if pred.shape != depth.shape:
        pred = cv2.resize(pred, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
    return pred.astype(np.float32), infer_ms


def render_video(args, input_path: Path, output_path: Path, summary_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type != "cuda":
        raise RuntimeError("TensorRT runner requires CUDA")

    with np.load(input_path, allow_pickle=False) as data:
        rgbs = data["rgb"]
        depths = data["depth"].astype(np.float32)
    if args.limit > 0:
        rgbs = rgbs[: args.limit]
        depths = depths[: args.limit]

    runner = TrtRunner(args.engine, device)
    for idx in range(min(args.warmup, len(rgbs))):
        _, ms = infer_one_trt(runner, rgbs[idx], depths[idx], args.max_side, device)
        print(f"warmup {input_path.name} frame={idx} trt_ms={ms:.2f}")

    valid_all = valid_depth(depths, args.max_depth)
    vals = depths[valid_all]
    lo, hi = np.percentile(vals, [2, 98]) if vals.size else (0.1, args.max_depth)

    h, w = depths.shape[1:]
    out_w = int(w * 4 * args.preview_scale)
    out_h = int(h * args.preview_scale)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {output_path}")

    rng = np.random.default_rng(20260603)
    infer_timings = []
    total_timings = []
    calib_timings = []
    stats = []

    for idx, (rgb, raw) in enumerate(zip(rgbs, depths)):
        pred, infer_ms = infer_one_trt(runner, rgb, raw, args.max_side, device)
        t0 = time.perf_counter()
        calibrated, frame_stats = calibrate_depth(raw, pred, args.max_depth, rng, args.max_samples, args.calib_downscale, args.eval_downscale)
        calib_ms = (time.perf_counter() - t0) * 1000.0
        total_ms = infer_ms + calib_ms
        infer_timings.append(infer_ms)
        calib_timings.append(calib_ms)
        total_timings.append(total_ms)
        frame_stats["infer_ms"] = infer_ms
        frame_stats["calib_ms"] = calib_ms
        frame_stats["frame_ms"] = total_ms
        frame_stats["frame_fps"] = 1000.0 / total_ms if total_ms > 0 else np.nan
        stats.append(frame_stats)

        raw_valid = valid_depth(raw, args.max_depth)
        calibrated_valid = valid_depth(calibrated, args.max_depth)
        eval_valid = raw_valid & calibrated_valid
        panels = [
            put_label(rgb, f"RGB | frame {idx:03d}"),
            put_label(colorize_depth(raw, raw_valid, lo, hi), f"Raw metric | valid {raw_valid.mean()*100:.1f}%"),
            put_label(
                colorize_depth(calibrated, calibrated_valid, lo, hi),
                "LingBot TRT FP16 calibrated | "
                f"FPS {frame_stats['frame_fps']:.1f} | infer {infer_ms:.1f}ms | a={frame_stats['scale']:.3f} b={frame_stats['bias']:.2f}\n"
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
            print(f"processed {input_path.name}: {idx + 1}/{len(rgbs)}")

    writer.release()
    infer_np = np.asarray(infer_timings, dtype=np.float64)
    total_np = np.asarray(total_timings, dtype=np.float64)
    calib_np = np.asarray(calib_timings, dtype=np.float64)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "frames": int(len(rgbs)),
        "max_side": args.max_side,
        "num_tokens": args.num_tokens,
        "engine": str(args.engine),
        "tensor_info": runner.tensor_info,
        "infer_ms_mean": float(infer_np.mean()),
        "infer_ms_p50": float(np.percentile(infer_np, 50)),
        "infer_ms_p95": float(np.percentile(infer_np, 95)),
        "infer_fps_mean": float(1000.0 / infer_np.mean()),
        "calib_ms_mean": float(calib_np.mean()),
        "total_ms_mean": float(total_np.mean()),
        "total_ms_p50": float(np.percentile(total_np, 50)),
        "total_ms_p95": float(np.percentile(total_np, 95)),
        "total_fps_mean": float(1000.0 / total_np.mean()),
        "eval_mae_m_mean": float(np.nanmean([s["eval_mae_m"] for s in stats])),
        "eval_medae_m_mean": float(np.nanmean([s["eval_medae_m"] for s in stats])),
        "eval_p90_m_mean": float(np.nanmean([s["eval_p90_m"] for s in stats])),
        "raw_valid_pct_mean": float(np.mean([s["raw_valid_pct"] for s in stats])),
        "calibrated_valid_pct_mean": float(np.mean([s["calibrated_valid_pct"] for s in stats])),
        "cuda_peak_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def static_shape_from_input(input_path: Path, max_side: int) -> tuple[int, int, int, int]:
    with np.load(input_path, allow_pickle=False) as data:
        rgb = data["rgb"][0]
        depth = data["depth"][0].astype(np.float32)
    rgb_in, depth_in, _ = resize_inputs(rgb, depth, max_side)
    return (1, 3, int(depth_in.shape[0]), int(depth_in.shape[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.home() / "lingbot-depth")
    parser.add_argument("--model-path", type=Path, default=Path.home() / "models" / "lingbot-depth-postrain-dc-vitl14-model.pt")
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--engine", type=Path, default=None)
    parser.add_argument("--trtexec", type=Path, default=Path("/usr/src/tensorrt/bin/trtexec"))
    parser.add_argument("--trtexec-log", type=Path, default=None)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/trt_fp16_tokens600"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--max-side", type=int, default=320)
    parser.add_argument("--num-tokens", type=int, default=600)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--preview-scale", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--calib-downscale", type=int, default=8)
    parser.add_argument("--eval-downscale", type=int, default=8)
    parser.add_argument("--error-max-cm", type=float, default=20.0)
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--force-engine", action="store_true")
    args = parser.parse_args()

    input_shape = static_shape_from_input(args.input[0], args.max_side)
    _, _, static_h, static_w = input_shape
    if args.onnx is None:
        args.onnx = Path(f"trt/lingbot_tokens{args.num_tokens}_{static_h}x{static_w}.onnx")
    if args.engine is None:
        args.engine = Path(f"trt/lingbot_tokens{args.num_tokens}_{static_h}x{static_w}_fp16.engine")
    if args.trtexec_log is None:
        args.trtexec_log = Path(f"trt/trtexec_tokens{args.num_tokens}_{static_h}x{static_w}_fp16.log")

    print(f"static export shape={input_shape} max_side={args.max_side} num_tokens={args.num_tokens}")
    all_summary = {
        "export": export_onnx(args, input_shape),
        "trtexec": run_trtexec(args),
        "videos": [],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in args.input:
        stem = input_path.stem.replace("synced_", "").replace("_first30", "")
        output_path = args.output_dir / f"lingbot_trt_fp16_calibrated_heatmap_{stem}_tokens{args.num_tokens}_maxside{args.max_side}_first30.mp4"
        summary_path = output_path.with_suffix(".json")
        all_summary["videos"].append(render_video(args, input_path, output_path, summary_path))

    run_summary = args.output_dir / f"summary_trt_fp16_tokens{args.num_tokens}_maxside{args.max_side}.json"
    run_summary.write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
    print(f"wrote summary: {run_summary}")


if __name__ == "__main__":
    main()
