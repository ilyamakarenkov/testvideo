#!/usr/bin/env python3
"""
Нарезка видео на N равномерно распределённых кадров и удаление фона
(остаётся только идущий человек на прозрачном фоне).

Сегментация: YOLOv8n-seg (ONNX, класс COCO "person") через onnxruntime.
Модель запускается на полном кадре и на нативных тайлах 640x640 в верхней
части кадра — так маленький/далёкий человек детектируется надёжнее.

Результат:
  output/frames/          исходные кадры (JPG)
  output/frames_no_bg/    те же кадры без фона, полный размер (PNG, прозрачность)
  output/person_cropped/  человек, обрезанный по bounding box маски (PNG)
  output/report.csv       соответствие файл -> номер кадра видео, время, bbox, conf

Использование:
  python scripts/extract_person_frames.py --video VID.mp4 --count 50 \
      --model models/yolov8n-seg.onnx --out output
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

PERSON_CLASS = 0  # COCO


class YoloSeg:
    """YOLOv8-seg ONNX (экспорт Ultralytics): output0 [1,116,8400], output1 [1,32,160,160]."""

    def __init__(self, model_path, conf=0.32, iou=0.45, imgsz=640):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.conf, self.iou, self.imgsz = conf, iou, imgsz

    def _letterbox(self, img):
        h, w = img.shape[:2]
        s = self.imgsz
        r = min(s / h, s / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        left, top = (s - nw) // 2, (s - nh) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, r, left, top

    def predict(self, img, conf=None):
        """Возвращает список детектов человека: {box:[x1,y1,x2,y2], conf, prob: float-маска HxW}."""
        conf = self.conf if conf is None else conf
        orig_h, orig_w = img.shape[:2]
        canvas, r, left, top = self._letterbox(img)
        inp = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out0, out1 = self.session.run(None, {self.input_name: inp})
        preds = out0[0].T                     # [8400, 116]
        protos = out1[0]                       # [32, 160, 160]
        boxes_xywh, scores, coeffs = preds[:, :4], preds[:, 4:84], preds[:, 84:]
        cls_id = np.argmax(scores, axis=1)
        person_score = scores[:, PERSON_CLASS]
        idx = np.where((cls_id == PERSON_CLASS) & (person_score > conf))[0]
        if len(idx) == 0:
            return []

        b = boxes_xywh[idx]
        xyxy = np.empty_like(b)
        xyxy[:, 0] = b[:, 0] - b[:, 2] / 2
        xyxy[:, 1] = b[:, 1] - b[:, 3] / 2
        xyxy[:, 2] = b[:, 0] + b[:, 2] / 2
        xyxy[:, 3] = b[:, 1] + b[:, 3] / 2
        confs = person_score[idx]
        rects = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in xyxy]
        nms = cv2.dnn.NMSBoxes(rects, confs.tolist(), conf, self.iou)
        if len(nms) == 0:
            return []
        nms = np.array(nms).flatten()

        s = self.imgsz
        nw, nh = int(round(orig_w * r)), int(round(orig_h * r))
        proto_flat = protos.reshape(32, -1)
        results = []
        for j in nms:
            k = idx[j]
            x1, y1, x2, y2 = (float(v) for v in xyxy[j])
            # мягкая маска: коэффициенты x прототипы -> sigmoid
            m_small = 1.0 / (1.0 + np.exp(-(coeffs[k] @ proto_flat)))
            m_small = m_small.reshape(160, 160)
            # обрезаем маску по боксу (в масштабе 160), как делает Ultralytics
            mx1, my1 = max(0, int(x1 / s * 160)), max(0, int(y1 / s * 160))
            mx2, my2 = min(160, int(x2 / s * 160 + 0.5)), min(160, int(y2 / s * 160 + 0.5))
            cropped = np.zeros_like(m_small)
            cropped[my1:my2, mx1:mx2] = m_small[my1:my2, mx1:mx2]
            m = cv2.resize(cropped, (s, s), interpolation=cv2.INTER_LINEAR)
            m = m[top:top + nh, left:left + nw]
            m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            ox1 = max(0.0, (x1 - left) / r)
            oy1 = max(0.0, (y1 - top) / r)
            ox2 = min(float(orig_w), (x2 - left) / r)
            oy2 = min(float(orig_h), (y2 - top) / r)
            results.append({
                "box": [int(round(v)) for v in (ox1, oy1, ox2, oy2)],
                "conf": float(confs[j]),
                "prob": m,
            })
        results.sort(key=lambda d: -d["conf"])
        return results


def views_for(img):
    """Полный кадр + нативные тайлы 640x640 по верхним углам (если кадр больше входа модели)."""
    h, w = img.shape[:2]
    views = [(0, 0, img)]
    if h > 640 or w > 640:
        tw, th = min(640, w), min(640, h)
        cand = {(0, 0), (w - tw, 0)}
        for x0, y0 in sorted(cand):
            views.append((x0, y0, img[y0:y0 + th, x0:x0 + tw]))
    return views


def detect_person(seg, img, conf):
    """Собирает детекты со всех видов, переводит в координаты полного кадра."""
    h, w = img.shape[:2]
    all_dets = []
    for x0, y0, view in views_for(img):
        for d in seg.predict(view, conf=conf):
            box = [d["box"][0] + x0, d["box"][1] + y0, d["box"][2] + x0, d["box"][3] + y0]
            bw, bh = box[2] - box[0], box[3] - box[1]
            if bh < 50 or bw < 15 or bw > bh * 1.2:   # человек — вертикальный объект
                continue
            prob = np.zeros((h, w), dtype=np.float32)
            prob[y0:y0 + view.shape[0], x0:x0 + view.shape[1]] = d["prob"]
            all_dets.append({"box": box, "conf": d["conf"], "prob": prob})
    return all_dets


def detect_person_zoom(seg, img, conf=0.12):
    """
    Fallback для кадров, где человек не найден обычным проходом (сильная окклюзия):
    сетка перекрывающихся квадратных зум-кропов (~78% меньшей стороны кадра).
    Человек занимает в кропе больше места -> детектор видит даже фрагменты (ноги).
    """
    h, w = img.shape[:2]
    size = max(320, int(min(w, h) * 0.78))
    xs = sorted(set(int(v) for v in np.linspace(0, max(0, w - size), 3)))
    ys = sorted(set(int(v) for v in np.linspace(0, max(0, h - size), 4)))
    all_dets = []
    for y0 in ys:
        for x0 in xs:
            view = img[y0:y0 + size, x0:x0 + size]
            for d in seg.predict(view, conf=conf):
                box = [d["box"][0] + x0, d["box"][1] + y0, d["box"][2] + x0, d["box"][3] + y0]
                bw, bh = box[2] - box[0], box[3] - box[1]
                if bh < 60 or bw < 15 or bw > bh * 1.2:
                    continue
                # слабые обрубки у самой границы кадра — обычно артефакты (ствол/столб)
                if (box[0] <= 1 or box[2] >= w - 1) and d["conf"] < 0.25:
                    continue
                prob = np.zeros((h, w), dtype=np.float32)
                prob[y0:y0 + view.shape[0], x0:x0 + view.shape[1]] = d["prob"]
                all_dets.append({"box": box, "conf": d["conf"], "prob": prob})
    return all_dets


def merge_person(all_dets):
    """
    Выбирает главного человека (якорь — детект с макс. conf) и объединяет
    маски детектов того же человека с разных видов (пересечение боксов/масок).
    Возвращает (prob-маска или None, объединённый bbox, conf якоря).
    """
    if not all_dets:
        return None, None, 0.0
    anchor = max(all_dets, key=lambda d: d["conf"])
    ax1, ay1, ax2, ay2 = anchor["box"]
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    prob = anchor["prob"].copy()
    box = list(anchor["box"])
    anchor_bin = anchor["prob"] > 0.5
    for d in all_dets:
        if d is anchor or d["conf"] < 0.5 * anchor["conf"]:
            continue  # не склеиваем сильно более слабые гипотезы (листва, столбы)
        x1, y1, x2, y2 = d["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        center_near = abs(cx - acx) < 0.75 * (ax2 - ax1 + x2 - x1) and abs(cy - acy) < 0.75 * (ay2 - ay1)
        d_bin = d["prob"] > 0.5
        overlap = np.logical_and(d_bin, anchor_bin).sum()
        mask_overlap = overlap > 0.05 * max(1, d_bin.sum())
        if center_near or mask_overlap:
            prob = np.maximum(prob, d["prob"])
            box = [min(box[0], x1), min(box[1], y1), max(box[2], x2), max(box[3], y2)]
    return prob, box, anchor["conf"]


def prob_to_alpha(prob, lo=0.35, hi=0.65, min_component=150):
    """Мягкий альфа-канал из prob-маски: smoothstep по краям + чистка мелких компонентов."""
    alpha = np.clip((prob - lo) / (hi - lo), 0.0, 1.0)
    alpha = alpha * alpha * (3 - 2 * alpha)                      # smoothstep
    binary = (prob > 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_component:
            keep[labels == i] = 1
    # альфу оставляем только рядом с сохранёнными компонентами (dilate — запас на мягкий край)
    keep_d = cv2.dilate(keep, np.ones((9, 9), np.uint8))
    alpha = alpha * keep_d
    alpha_u8 = (alpha * 255).astype(np.uint8)
    alpha_u8 = cv2.GaussianBlur(alpha_u8, (3, 3), 0)              # лёгкое сглаживание края
    return alpha_u8


def bbox_of_alpha(alpha, thresh=10, pad=12):
    ys, xs = np.where(alpha > thresh)
    if len(xs) == 0:
        return None
    h, w = alpha.shape
    return [max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(w, int(xs.max()) + pad + 1), min(h, int(ys.max()) + pad + 1)]


def make_preview(cropped_dir, count, out_path, cell_h=260, cell_w=150, cols=10):
    """Контактный лист вырезанного человека на шахматной подложке (видна прозрачность)."""
    def checker(h, w, s=16):
        yy, xx = np.mgrid[0:h, 0:w]
        c = np.where(((xx // s + yy // s) % 2 == 0), 200, 130).astype(np.uint8)
        return np.repeat(c[:, :, None], 3, axis=2)

    cells = []
    for i in range(1, count + 1):
        cell = np.full((cell_h + 24, cell_w, 3), 255, np.uint8)
        cv2.putText(cell, f"{i:03d}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        path = os.path.join(cropped_dir, f"frame_{i:03d}.png")
        if os.path.exists(path):
            rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            a = rgba[:, :, 3:4].astype(np.float32) / 255.0
            img = (rgba[:, :, :3].astype(np.float32) * a +
                   checker(*rgba.shape[:2]).astype(np.float32) * (1 - a)).astype(np.uint8)
            scale = cell_h / img.shape[0]
            img = cv2.resize(img, (max(8, int(img.shape[1] * scale)), cell_h),
                             interpolation=cv2.INTER_AREA)
        else:  # человек на кадре полностью скрыт (дерево/столб)
            img = np.full((cell_h, 90, 3), (0, 0, 200), np.uint8)
            cv2.putText(img, "hidden", (8, cell_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        w = min(cell_w, img.shape[1])
        x = (cell_w - w) // 2
        cell[24:, x:x + w] = img[:, :w]
        cells.append(cell)
    while len(cells) % cols:
        cells.append(np.full_like(cells[0], 255))
    sheet = np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])
    cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="VID.mp4")
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--model", default="models/yolov8n-seg.onnx")
    ap.add_argument("--out", default="output")
    ap.add_argument("--conf", type=float, default=0.32)
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        sys.exit(f"Не найден видео файл: {args.video}")
    if not os.path.isfile(args.model):
        sys.exit(f"Не найдена модель: {args.model}")

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if total <= 0:
        sys.exit("Не удалось прочитать видео")
    indices = np.unique(np.linspace(0, total - 1, args.count).round().astype(int))
    print(f"Видео: {args.video}  кадров={total} fps={fps:.2f}  нарезаем={len(indices)}")

    dirs = {k: os.path.join(args.out, k) for k in ("frames", "frames_no_bg", "person_cropped")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    seg = YoloSeg(args.model)
    rows, found = [], 0
    for n_out, frame_idx in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            print(f"[{n_out:02d}] кадр {frame_idx}: НЕ ПРОЧИТАН")
            continue
        stem = f"frame_{n_out:03d}"
        cv2.imwrite(os.path.join(dirs["frames"], stem + ".jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

        prob, box, conf = merge_person(detect_person(seg, frame, args.conf))
        tag = ""
        if prob is None:      # второй проход с пониженным порогом
            prob, box, conf = merge_person(detect_person(seg, frame, 0.20))
            tag = "_low_conf" if prob is not None else ""
        if prob is None:      # третий проход: зум-тайлы (видимые фрагменты при окклюзии)
            prob, box, conf = merge_person(detect_person_zoom(seg, frame, 0.12))
            tag = "_fallback_zoom" if prob is not None else ""

        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        crop_bbox = ""
        if prob is None:
            alpha = np.zeros(frame.shape[:2], dtype=np.uint8)
            status = "not_found"
        else:
            found += 1
            alpha = prob_to_alpha(prob)
            status = "ok" + tag
        rgba[:, :, 3] = alpha
        rgba[alpha == 0, :3] = 0   # не храним картинку под прозрачностью -> PNG в разы меньше
        if prob is not None:
            cb = bbox_of_alpha(alpha)
            if cb is not None:
                crop_bbox = "x".join(map(str, cb))
                cv2.imwrite(os.path.join(dirs["person_cropped"], stem + ".png"),
                            rgba[cb[1]:cb[3], cb[0]:cb[2]])
        cv2.imwrite(os.path.join(dirs["frames_no_bg"], stem + ".png"), rgba)

        t = frame_idx / fps if fps else 0
        print(f"[{n_out:02d}/{len(indices)}] src_frame={int(frame_idx):4d} t={t:5.2f}s "
              f"status={status} conf={conf:.3f} box={box}")
        rows.append({"file": stem, "src_frame": int(frame_idx), "time_sec": f"{t:.2f}",
                     "status": status, "conf": f"{conf:.3f}",
                     "box_xyxy": ("x".join(map(str, box)) if box else ""),
                     "crop_xyxy": crop_bbox})
    cap.release()

    with open(os.path.join(args.out, "report.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "src_frame", "time_sec", "status", "conf", "box_xyxy", "crop_xyxy"])
        w.writeheader()
        w.writerows(rows)
    preview_path = os.path.join(args.out, "preview_person_cropped.jpg")
    make_preview(dirs["person_cropped"], len(rows), preview_path)
    print(f"\nГотово: кадров {len(rows)}, человек найден на {found}, не найден на {len(rows) - found}")
    print(f"Результаты: {args.out}/frames, {args.out}/frames_no_bg, {args.out}/person_cropped, "
          f"{args.out}/report.csv, {preview_path}")


if __name__ == "__main__":
    main()
