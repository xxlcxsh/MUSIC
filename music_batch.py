#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тонкая обёртка для пакетной обработки (совместимость)."""

import sys

from music import batch_process, main as music_main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Используйте: python music.py --batch audio [--output file.csv] [--visual]")
        sys.exit(0)
    # Поддержка старого интерфейса: python music_batch.py --audio-dir audio
    if "--audio-dir" in sys.argv:
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("--audio-dir", default="audio")
        p.add_argument("--output", default=None)
        p.add_argument("--speed-of-sound", type=float, default=343.0)
        p.add_argument("--block-start", type=int, default=0)
        p.add_argument("--block-len", type=int, default=88200)
        p.add_argument("--visual", action="store_true")
        a = p.parse_args()
        batch_process(
            a.audio_dir,
            a.output,
            c=a.speed_of_sound,
            block_start=a.block_start,
            block_len=a.block_len,
            visual=a.visual,
        )
    else:
        music_main()
