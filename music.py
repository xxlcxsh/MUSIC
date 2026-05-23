#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI для широкополосного near-field MUSIC.
Пример:
  python music.py audio/A1_CH_10_20.aup
  python music.py --batch audio --output results.csv
  python music.py audio/A1_CH_10_20.aup --visual --visual-dir viz_out
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from datetime import datetime

from music_core import (
    C_SOUND_DEFAULT,
    FS_DEFAULT,
    load_audacity_or_wav,
    music_localization,
    parse_ground_truth_from_filename,
)
from music_visualize import save_all_visualizations


def process_file(
    audio_path: str,
    *,
    c: float = C_SOUND_DEFAULT,
    block_start: int = 0,
    block_len: int = 44100,
    num_sources: int = 1,
    visual: bool = False,
    visual_dir: str | None = None,
    full_3d: bool = False,
) -> dict:
    x, sr = load_audacity_or_wav(audio_path)
    end = min(block_start + block_len, x.shape[1])
    segment = x[:, block_start:end]
    if segment.shape[1] < 1024:
        raise ValueError("Блок слишком короткий (<1024 отсчётов)")

    result = music_localization(
        segment,
        sr,
        num_sources=num_sources,
        c=c,
        return_diagnostics=visual,
        full_3d=full_3d,
    )

    gt = parse_ground_truth_from_filename(audio_path)
    if visual:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        out = visual_dir or os.path.join("visualizations", base)
        save_all_visualizations(result, out, title_prefix=base, ground_truth=gt)

    row = {
        "filename": os.path.basename(audio_path),
        "azimuth_deg": round(result.azimuth_deg, 2),
        "elevation_deg": round(result.elevation_deg, 2),
        "distance_m": round(result.distance_m, 3),
        "gt_azimuth_deg": round(gt[0], 2) if gt else "",
        "gt_distance_m": round(gt[1], 3) if gt else "",
        "error": "",
    }
    return row


def batch_process(
    audio_dir: str,
    output_csv: str | None = None,
    *,
    c: float = C_SOUND_DEFAULT,
    block_start: int = 0,
    block_len: int = 44100,
    num_sources: int = 1,
    visual: bool = False,
    visual_root: str = "visualizations",
    full_3d: bool = False,
) -> list[dict]:
    patterns = [
        os.path.join(audio_dir, "*.aup"),
        os.path.join(audio_dir, "*.wav"),
        os.path.join(audio_dir, "*.flac"),
    ]
    files: list[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(f"В {audio_dir} не найдено .aup/.wav/.flac файлов")

    if output_csv is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"music_results_{ts}.csv"

    results: list[dict] = []
    print(f"Batch: {len(files)} files -> {output_csv}")

    for i, path in enumerate(files, 1):
        name = os.path.basename(path)
        print(f"[{i}/{len(files)}] {name}")
        try:
            vdir = os.path.join(visual_root, os.path.splitext(name)[0]) if visual else None
            row = process_file(
                path,
                c=c,
                block_start=block_start,
                block_len=block_len,
                num_sources=num_sources,
                visual=visual,
                visual_dir=vdir,
                full_3d=full_3d,
            )
            print(
                f"    azimuth={row['azimuth_deg']} deg, elevation={row['elevation_deg']} deg, "
                f"distance={row['distance_m']} m"
            )
        except Exception as exc:
            row = {
                "filename": name,
                "azimuth_deg": "",
                "elevation_deg": "",
                "distance_m": "",
                "gt_azimuth_deg": "",
                "gt_distance_m": "",
                "error": str(exc)[:200],
            }
            print(f"    Ошибка: {exc}")

        results.append(row)

    fieldnames = [
        "filename",
        "azimuth_deg",
        "elevation_deg",
        "distance_m",
        "gt_azimuth_deg",
        "gt_distance_m",
        "error",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    ok = sum(1 for r in results if not r["error"])
    print(f"Готово: {ok}/{len(files)} успешно. CSV: {output_csv}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Широкополосный near-field MUSIC (8-канальная решётка)"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Путь к .aup / многоканальному .wav",
    )
    parser.add_argument(
        "--batch",
        type=str,
        metavar="DIR",
        help="Пакетная обработка всех файлов в каталоге",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV для пакетного режима",
    )
    parser.add_argument("--speed-of-sound", type=float, default=C_SOUND_DEFAULT, metavar="C")
    parser.add_argument("--block-start", type=int, default=0)
    parser.add_argument("--block-len", type=int, default=44100)
    parser.add_argument("--num-sources", type=int, default=1)
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Сохранить подробные визуализации в отдельную директорию",
    )
    parser.add_argument(
        "--visual-dir",
        type=str,
        default=None,
        help="Каталог визуализаций (одиночный файл)",
    )
    parser.add_argument(
        "--visual-root",
        type=str,
        default="visualizations",
        help="Корень визуализаций в пакетном режиме",
    )
    parser.add_argument(
        "--full-3d",
        action="store_true",
        help="Полная 3D-сетка (theta, phi, d) по instruction.md вместо планарной 2D",
    )

    args = parser.parse_args()

    if args.batch:
        if not os.path.isdir(args.batch):
            print(f"Директория не найдена: {args.batch}", file=sys.stderr)
            sys.exit(1)
        try:
            batch_process(
                args.batch,
                args.output,
                c=args.speed_of_sound,
                block_start=args.block_start,
                block_len=args.block_len,
                num_sources=args.num_sources,
                visual=args.visual,
                visual_root=args.visual_root,
                full_3d=args.full_3d,
            )
        except Exception as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    if not args.audio_file:
        parser.error("Укажите audio_file или --batch DIR")

    try:
        row = process_file(
            args.audio_file,
            c=args.speed_of_sound,
            block_start=args.block_start,
            block_len=args.block_len,
            num_sources=args.num_sources,
            visual=args.visual,
            visual_dir=args.visual_dir,
            full_3d=args.full_3d,
        )
        print("\n" + "=" * 60)
        print(" LOCALIZATION RESULT")
        print(f"  Направление звука: {row['azimuth_deg']}°")
        print(f"  (0° = правее центра массива, против часовой стрелки)")
        print(f"  Azimuth (deg):    {row['azimuth_deg']}")
        print(f"  Elevation (deg):  {row['elevation_deg']}")
        print(f"  Distance (m):     {row['distance_m']}")
        if row["gt_azimuth_deg"] != "":
            print(f"  Ground truth:     azimuth={row['gt_azimuth_deg']}, distance={row['gt_distance_m']} m")
        print("=" * 60 + "\n")
    except Exception as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
