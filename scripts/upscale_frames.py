#!/usr/bin/env python3
"""
AI upscale изображений в 4 раза моделью RealESRGAN x4plus (ONNX).

Большие изображения обрабатываются перекрывающимися тайлами 256x256,
результаты сшиваются с линейным cross-fade в зонах перекрытия, поэтому швов нет.
Модель не хранится в git из-за размера (67 МБ), см. раздел про upscale в README.md.

Пример:
  python scripts/upscale_frames.py \
      output/frames/frame_001.jpg output/frames/frame_002.jpg output/frames/frame_007.jpg \
      --model models/realesrgan-x4plus.onnx --out output/upscaled_4x
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


def tile_starts(length, tile, overlap):
    """Равномерно распределённые позиции тайлов, полностью покрывающие [0, length]."""
    if length <= tile:
        return [0]
    stride = tile - overlap
    count = int(np.ceil((length - tile) / stride)) + 1
    return sorted(set(int(v) for v in np.linspace(0, length - tile, count).round()))


def axis_weights(starts, index, tile, scale):
    """
    1D-веса тайла: линейный ramp по всей зоне перекрытия с соседними тайлами.
    У глобальных краёв изображения вес остаётся 1.
    """
    size = tile * scale
    w = np.ones(size, dtype=np.float32)
    s = starts[index]
    if index > 0:
        n = max(1, (starts[index - 1] + tile - s) * scale)
        w[:n] = (np.arange(n, dtype=np.float32) + 0.5) / n
    if index < len(starts) - 1:
        n = max(1, (s + tile - starts[index + 1]) * scale)
        w[-n:] = (n - np.arange(n, dtype=np.float32) - 0.5) / n
    return w


class RealEsrganUpscaler:
    def __init__(self, model_path, scale=4, tile=256, overlap=32, threads=2):
        if not os.path.isfile(model_path):
            sys.exit(f"Не найдена модель: {model_path}\n"
                     "Как её получить — см. раздел 'Upscale' в README.md")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.scale, self.tile, self.overlap = scale, tile, overlap

    def _run_tile(self, tile_rgb_f32):
        x = tile_rgb_f32.transpose(2, 0, 1)[None]
        y = self.session.run(None, {self.input_name: x})[0][0]
        return np.clip(y.transpose(1, 2, 0), 0.0, 1.0)

    def upscale(self, img_rgb_u8, progress_prefix=""):
        h, w = img_rgb_u8.shape[:2]
        s, t = self.scale, self.tile
        # паддинг отражением, если сторона меньше тайла
        pad_h, pad_w = max(0, t - h), max(0, t - w)
        img = cv2.copyMakeBorder(img_rgb_u8, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        ph, pw = img.shape[:2]
        img_f = img.astype(np.float32) / 255.0

        xs = tile_starts(pw, t, self.overlap)
        ys = tile_starts(ph, t, self.overlap)
        out = np.zeros((ph * s, pw * s, 3), dtype=np.float32)
        wsum = np.zeros((ph * s, pw * s, 1), dtype=np.float32)

        total = len(xs) * len(ys)
        done = 0
        for iy, y0 in enumerate(ys):
            wy = axis_weights(ys, iy, t, s)[:, None]
            for ix, x0 in enumerate(xs):
                tic = time.time()
                tile = self._run_tile(img_f[y0:y0 + t, x0:x0 + t])
                if tile.shape[:2] != (t * s, t * s):
                    sys.exit(f"Модель вернула {tile.shape}, ожидался масштаб x{s}")
                weight = (wy * axis_weights(xs, ix, t, s)[None, :])[:, :, None]
                region = np.s_[y0 * s:(y0 + t) * s, x0 * s:(x0 + t) * s]
                out[region] += tile * weight
                wsum[region] += weight
                done += 1
                print(f"{progress_prefix}тайл {done:2d}/{total} "
                      f"(x={x0}, y={y0}) {time.time() - tic:.1f} c", flush=True)
        out /= np.maximum(wsum, 1e-6)
        out = (out[:h * s, :w * s] * 255.0).round().astype(np.uint8)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="входные изображения")
    ap.add_argument("--model", default="models/realesrgan-x4plus.onnx")
    ap.add_argument("--out", default="output/upscaled_4x")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--quality", type=int, default=95, help="качество JPEG")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    up = RealEsrganUpscaler(args.model, args.scale, args.tile, args.overlap, args.threads)

    for path in args.inputs:
        if not os.path.isfile(path):
            sys.exit(f"Не найден входной файл: {path}")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            sys.exit(f"Не удалось прочитать изображение: {path}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        stem = os.path.splitext(os.path.basename(path))[0]
        print(f"\n{path}: {rgb.shape[1]}x{rgb.shape[0]} -> "
              f"{rgb.shape[1] * args.scale}x{rgb.shape[0] * args.scale}", flush=True)
        tic = time.time()
        result = up.upscale(rgb, progress_prefix="  ")
        ext = ".png" if path.lower().endswith(".png") else ".jpg"
        dst = os.path.join(args.out, f"{stem}_{args.scale}x{ext}")
        pil = Image.fromarray(result)
        if ext == ".png":
            pil.save(dst, optimize=True)
        else:
            pil.save(dst, quality=args.quality, subsampling=0, optimize=True)
        print(f"  -> {dst}  {result.shape[1]}x{result.shape[0]}  "
              f"{os.path.getsize(dst) / 1e6:.1f} МБ  за {time.time() - tic:.0f} c", flush=True)


if __name__ == "__main__":
    main()
