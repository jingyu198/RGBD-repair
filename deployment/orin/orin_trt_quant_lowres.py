from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from orin_lingbot_trt_fp16 import (
    TrtRunner,
    calibrate_depth,
    colorize_depth,
    colorize_error,
    export_onnx,
    prepare_tensors,
    resize_inputs,
    run_trtexec,
    static_shape_from_input,
    valid_depth,
)


def parse_trtexec_stats(text: str) -> dict[str, float | None]:
    throughput = None
    gpu_compute_ms = None
    m = re.search(r"Throughput:\s*([0-9.]+)\s*qps", text)
    if m:
        throughput = float(m.group(1))
    m = re.search(r"GPU Compute Time:\s*min\s*=\s*[0-9.]+\s*ms,\s*max\s*=\s*[0-9.]+\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms", text)
    if m:
        gpu_compute_ms = float(m.group(1))
    return {"trtexec_throughput_fps": throughput, "trtexec_gpu_compute_ms": gpu_compute_ms}


def run_engine_benchmark(trtexec: Path, engine: Path, log_path: Path) -> dict:
    cmd = [
        str(trtexec),
        f"--loadEngine={engine}",
        "--warmUp=500",
        "--duration=5",
        "--avgRuns=100",
    ]
    print("running:", " ".join(cmd))
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout[-5000:])
    if proc.returncode != 0:
        raise RuntimeError(f"trtexec benchmark failed with code {proc.returncode}; see {log_path}")
    return parse_trtexec_stats(proc.stdout)


class TorchInt8Calibrator:
    def __init__(self, trt, samples: list[dict[str, np.ndarray]], cache_path: Path, device: torch.device):
        class _Calibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer

            def get_batch_size(self):
                return 1

            def get_batch(self, names):
                outer = self.outer
                if outer.index >= len(outer.samples):
                    return None
                sample = outer.samples[outer.index]
                outer.index += 1
                image = torch.tensor(sample["image"], dtype=torch.float32, device=outer.device).contiguous()
                depth = torch.tensor(sample["depth"], dtype=torch.float32, device=outer.device).contiguous()
                outer.current = {"image": image, "depth": depth}
                return [int(outer.current[name].data_ptr()) for name in names]

            def read_calibration_cache(self):
                if self.outer.cache_path.exists() and not self.outer.force:
                    return self.outer.cache_path.read_bytes()
                return None

            def write_calibration_cache(self, cache):
                self.outer.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.outer.cache_path.write_bytes(bytes(cache))

        self.samples = samples
        self.cache_path = cache_path
        self.device = device
        self.index = 0
        self.current = {}
        self.force = False
        self.impl = _Calibrator(self)


def collect_calibration_samples(input_paths: list[Path], max_side: int, limit_per_input: int) -> list[dict[str, np.ndarray]]:
    samples: list[dict[str, np.ndarray]] = []
    for input_path in input_paths:
        with np.load(input_path, allow_pickle=False) as data:
            rgbs = data["rgb"]
            depths = data["depth"].astype(np.float32)
        if limit_per_input > 0:
            rgbs = rgbs[:limit_per_input]
            depths = depths[:limit_per_input]
        for rgb, depth in zip(rgbs, depths):
            rgb_small, depth_small, _ = resize_inputs(rgb, depth, max_side)
            image_t, depth_t = prepare_tensors(rgb_small, depth_small, torch.device("cpu"))
            samples.append({"image": image_t.numpy(), "depth": depth_t.numpy()})
    return samples


def build_int8_engine(args, samples: list[dict[str, np.ndarray]], device: torch.device) -> dict:
    import tensorrt as trt

    if args.int8_engine.exists() and not args.force_int8:
        print(f"INT8 engine exists: {args.int8_engine}")
        stats = run_engine_benchmark(args.trtexec, args.int8_engine, args.int8_benchmark_log)
        return {"skipped": True, "engine": str(args.int8_engine), **stats}

    args.int8_engine.parent.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(args.onnx.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb * 1024 * 1024)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    calibrator = TorchInt8Calibrator(trt, samples, args.calib_cache, device)
    calibrator.force = args.force_calib
    config.int8_calibrator = calibrator.impl

    start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_ms = (time.perf_counter() - start) * 1000.0
    if serialized is None:
        raise RuntimeError("INT8 TensorRT build failed")
    args.int8_engine.write_bytes(bytes(serialized))
    print(f"built INT8 engine: {args.int8_engine} ({args.int8_engine.stat().st_size / (1024**2):.1f} MB)")
    stats = run_engine_benchmark(args.trtexec, args.int8_engine, args.int8_benchmark_log)
    return {
        "skipped": False,
        "engine": str(args.int8_engine),
        "build_ms": build_ms,
        "calibration_samples": len(samples),
        "calib_cache": str(args.calib_cache),
        **stats,
    }


def stem_from_input(input_path: Path) -> str:
    stem = input_path.stem.replace("synced_", "").replace("_first30", "")
    if stem == "synced":
        stem = input_path.parent.name.replace("input_", "")
        if stem == "input":
            stem = "115349"
    return stem


def cv2_writer(path: Path, width: int, height: int, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {path}")
    return writer


def render_mode(args, mode: str, engine_path: Path, input_path: Path, device: torch.device) -> dict:
    runner = TrtRunner(engine_path, device)
    with np.load(input_path, allow_pickle=False) as data:
        rgbs = data["rgb"]
        depths = data["depth"].astype(np.float32)
    if args.limit > 0:
        rgbs = rgbs[: args.limit]
        depths = depths[: args.limit]

    small_rgbs = []
    small_depths = []
    for rgb, depth in zip(rgbs, depths):
        rgb_small, depth_small, _ = resize_inputs(rgb, depth, args.max_side)
        small_rgbs.append(rgb_small)
        small_depths.append(depth_small)
    small_rgbs = np.asarray(small_rgbs)
    small_depths = np.asarray(small_depths, dtype=np.float32)
    h, w = small_depths.shape[1:]

    for idx in range(min(args.warmup, len(small_rgbs))):
        image_t, depth_t = prepare_tensors(small_rgbs[idx], small_depths[idx], device)
        _, ms = runner.infer(image_t, depth_t)
        print(f"warmup {mode} {input_path.name} frame={idx} trt_ms={ms:.2f}")

    valid_all = valid_depth(small_depths, args.max_depth)
    vals = small_depths[valid_all]
    lo, hi = np.percentile(vals, [2, 98]) if vals.size else (0.1, args.max_depth)

    stem = stem_from_input(input_path)
    native_path = args.output_dir / mode / f"lingbot_lowres64_{mode}_{stem}_tokens{args.num_tokens}_native_256x36.mp4"
    viewer_path = args.output_dir / mode / f"lingbot_lowres64_{mode}_{stem}_tokens{args.num_tokens}_viewer_2560x360.mp4"
    native_writer = cv2_writer(native_path, w * 4, h, args.fps)
    viewer_writer = cv2_writer(viewer_path, w * 4 * args.viewer_scale, h * args.viewer_scale, args.fps)

    rng = np.random.default_rng(20260603)
    infer_ms_all = []
    total_ms_all = []
    stats = []

    for idx, (rgb_small, raw_small) in enumerate(zip(small_rgbs, small_depths)):
        image_t, depth_t = prepare_tensors(rgb_small, raw_small, device)
        pred, infer_ms = runner.infer(image_t, depth_t)
        pred = pred.squeeze(0).astype(np.float32)
        t0 = time.perf_counter()
        calibrated, frame_stats = calibrate_depth(
            raw_small,
            pred,
            args.max_depth,
            rng,
            args.max_samples,
            1,
            1,
        )
        calib_ms = (time.perf_counter() - t0) * 1000.0
        total_ms = infer_ms + calib_ms
        infer_ms_all.append(infer_ms)
        total_ms_all.append(total_ms)
        stats.append(frame_stats)

        raw_valid = valid_depth(raw_small, args.max_depth)
        calibrated_valid = valid_depth(calibrated, args.max_depth)
        eval_valid = raw_valid & calibrated_valid
        panels = [
            rgb_small,
            colorize_depth(raw_small, raw_valid, lo, hi),
            colorize_depth(calibrated, calibrated_valid, lo, hi),
            colorize_error(raw_small, calibrated, eval_valid, args.error_max_cm / 100.0),
        ]
        native = np.concatenate(panels, axis=1).astype(np.uint8)
        viewer = cv2.resize(native, (native.shape[1] * args.viewer_scale, native.shape[0] * args.viewer_scale), interpolation=cv2.INTER_NEAREST)
        native_writer.write(cv2.cvtColor(native, cv2.COLOR_RGB2BGR))
        viewer_writer.write(cv2.cvtColor(viewer, cv2.COLOR_RGB2BGR))

    native_writer.release()
    viewer_writer.release()

    infer_np = np.asarray(infer_ms_all, dtype=np.float64)
    total_np = np.asarray(total_ms_all, dtype=np.float64)
    summary = {
        "mode": mode,
        "input": str(input_path),
        "engine": str(engine_path),
        "native_output": str(native_path),
        "viewer_output": str(viewer_path),
        "frames": int(len(small_rgbs)),
        "panel_width": int(w),
        "panel_height": int(h),
        "native_width": int(w * 4),
        "native_height": int(h),
        "viewer_width": int(w * 4 * args.viewer_scale),
        "viewer_height": int(h * args.viewer_scale),
        "max_side": args.max_side,
        "num_tokens": args.num_tokens,
        "infer_ms_mean": float(infer_np.mean()),
        "infer_fps_mean": float(1000.0 / infer_np.mean()),
        "total_ms_mean": float(total_np.mean()),
        "total_fps_mean": float(1000.0 / total_np.mean()),
        "eval_mae_m_mean": float(np.nanmean([s["eval_mae_m"] for s in stats])),
        "eval_medae_m_mean": float(np.nanmean([s["eval_medae_m"] for s in stats])),
        "eval_p90_m_mean": float(np.nanmean([s["eval_p90_m"] for s in stats])),
        "raw_valid_pct_mean": float(np.mean([s["raw_valid_pct"] for s in stats])),
        "calibrated_valid_pct_mean": float(np.mean([s["calibrated_valid_pct"] for s in stats])),
        "cuda_peak_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
    }
    summary_path = args.output_dir / mode / f"lingbot_lowres64_{mode}_{stem}_tokens{args.num_tokens}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.home() / "lingbot-depth")
    parser.add_argument("--model-path", type=Path, default=Path.home() / "models" / "lingbot-depth-postrain-dc-vitl14-model.pt")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/quant_lowres64_tokens100"))
    parser.add_argument("--trtexec", type=Path, default=Path("/usr/src/tensorrt/bin/trtexec"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-side", type=int, default=64)
    parser.add_argument("--num-tokens", type=int, default=100)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--viewer-scale", type=int, default=10)
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--error-max-cm", type=float, default=20.0)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--calib-frames-per-input", type=int, default=30)
    parser.add_argument("--workspace-mb", type=int, default=4096)
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--force-fp16", action="store_true")
    parser.add_argument("--force-int8", action="store_true")
    parser.add_argument("--force-calib", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This experiment requires CUDA")

    input_shape = static_shape_from_input(args.input[0], args.max_side)
    _, _, static_h, static_w = input_shape
    trt_dir = Path("trt")
    args.onnx = trt_dir / f"lingbot_tokens{args.num_tokens}_{static_h}x{static_w}.onnx"
    args.engine = trt_dir / f"lingbot_tokens{args.num_tokens}_{static_h}x{static_w}_fp16.engine"
    args.int8_engine = trt_dir / f"lingbot_tokens{args.num_tokens}_{static_h}x{static_w}_int8.engine"
    args.trtexec_log = trt_dir / f"trtexec_tokens{args.num_tokens}_{static_h}x{static_w}_fp16.log"
    args.int8_benchmark_log = trt_dir / f"trtexec_tokens{args.num_tokens}_{static_h}x{static_w}_int8.log"
    args.calib_cache = trt_dir / f"lingbot_tokens{args.num_tokens}_{static_h}x{static_w}_int8_calib.cache"
    args.force_engine = args.force_fp16

    print(f"device={device} gpu={torch.cuda.get_device_name(0)}")
    print(f"static_shape={input_shape}")

    export_info = export_onnx(args, input_shape)
    fp16_info = run_trtexec(args)
    samples = collect_calibration_samples(args.input, args.max_side, args.calib_frames_per_input)
    int8_info = build_int8_engine(args, samples, device)

    videos = {"fp16": [], "int8": []}
    for input_path in args.input:
        videos["fp16"].append(render_mode(args, "fp16", args.engine, input_path, device))
        videos["int8"].append(render_mode(args, "int8", args.int8_engine, input_path, device))

    summary = {
        "export": export_info,
        "fp16_engine": fp16_info,
        "int8_engine": int8_info,
        "videos": videos,
    }
    summary_path = args.output_dir / f"summary_quant_lowres64_tokens{args.num_tokens}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
