#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пакетная обработка всех .aup файлов в директории audio/
Результаты сохраняются в CSV файл.
"""

import argparse
import os
import sys
import csv
import glob
from datetime import datetime
import numpy as np

# Импортируем функции из music.py
from music import run, load_audacity_or_wav, preprocess_signals, compute_nearfield_music
from music import NUM_MICS, FS_DEFAULT, MIC_X, MIC_Y, THETA_DEG, R_METERS, FREQ_RANGE

def process_single_file(aup_path, c=343.0, block_start=0, block_len=88200, verbose=True):
    """
    Обрабатывает один файл и возвращает результат.
    Возвращает: (theta, distance, error_message или None)
    """
    try:
        if verbose:
            print(f"\n📁 Обработка: {os.path.basename(aup_path)}")
        
        # Загрузка данных
        X, sr = load_audacity_or_wav(aup_path)
        current_fs = sr
        
        # Берём первый блок
        end = min(block_start + block_len, X.shape[1])
        X_seg = X[:, block_start:end]
        
        if X_seg.shape[1] < 1024:
            return None, None, "Блок слишком короткий (<1024 отсчёта)"
        
        # Предобработка и расчёт MUSIC
        X_proc = preprocess_signals(X_seg, current_fs)
        P_dB, P_lin = compute_nearfield_music(
            X_proc, current_fs, c, MIC_X, MIC_Y, THETA_DEG, R_METERS, FREQ_RANGE
        )
        
        # Поиск пика
        max_idx = np.unravel_index(np.argmax(P_lin), P_lin.shape)
        est_r = R_METERS[max_idx[0]]
        est_theta = THETA_DEG[max_idx[1]]
        
        return est_theta, est_r, None
        
    except Exception as e:
        return None, None, str(e)


def batch_process(audio_dir, output_csv=None, c=343.0, block_start=0, block_len=88200):
    """
    Обрабатывает все .aup файлы в директории.
    """
    # Поиск всех .aup файлов
    aup_files = glob.glob(os.path.join(audio_dir, "*.aup"))
    if not aup_files:
        print(f"❌ В директории {audio_dir} не найдено .aup файлов")
        return
    
    print(f"{'='*70}")
    print(f"  ПАКЕТНАЯ ОБРАБОТКА {len(aup_files)} ФАЙЛОВ")
    print(f"{'='*70}")
    print(f"📂 Директория: {audio_dir}")
    print(f"📄 Найдено файлов: {len(aup_files)}")
    print(f"⚙️  Параметры: c={c} м/с, block=[{block_start}:{block_start+block_len}]")
    print(f"{'='*70}\n")
    
    # Генерация имени CSV файла
    if output_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"music_results_{timestamp}.csv"
    
    results = []
    success_count = 0
    error_count = 0
    
    for i, aup_path in enumerate(sorted(aup_files), 1):
        filename = os.path.basename(aup_path)
        print(f"[{i}/{len(aup_files)}] {filename}")
        
        theta, distance, error = process_single_file(
            aup_path, c=c, block_start=block_start, block_len=block_len, verbose=False
        )
        
        if error is None:
            results.append({
                'filename': filename,
                'theta_deg': round(theta, 2),
                'distance_m': round(distance, 3),
                'error': ''
            })
            success_count += 1
            print(f"    ✅ θ={theta:.1f}°, r={distance:.2f} м")
        else:
            results.append({
                'filename': filename,
                'theta_deg': '',
                'distance_m': '',
                'error': error[:100]  # Обрезаем длинные ошибки
            })
            error_count += 1
            print(f"    ❌ Ошибка: {error[:60]}...")
    
    # Запись в CSV
    print(f"\n{'='*70}")
    print(f"📊 Результаты:")
    print(f"   Успешно: {success_count}")
    print(f"   Ошибки: {error_count}")
    print(f"   Всего: {len(aup_files)}")
    print(f"{'='*70}\n")
    
    print(f"💾 Сохранение результатов в {output_csv}...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'theta_deg', 'distance_m', 'error']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Готово! Результаты сохранены в {output_csv}")
    print(f"{'='*70}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Пакетная обработка всех .aup файлов в директории audio/"
    )
    parser.add_argument(
        '--audio-dir', 
        type=str, 
        default='audio',
        help='Директория с .aup файлами (по умолчанию: audio)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Имя выходного CSV файла (по умолчанию: music_results_YYYYMMDD_HHMMSS.csv)'
    )
    parser.add_argument(
        '--speed-of-sound', 
        type=float, 
        default=343.0, 
        metavar='C',
        help='Скорость звука (м/с)'
    )
    parser.add_argument(
        '--block-start', 
        type=int, 
        default=0,
        help='Начальный отсчёт блока'
    )
    parser.add_argument(
        '--block-len', 
        type=int, 
        default=88200,
        help='Длина блока в отсчётах (по умолчанию 2 сек)'
    )
    
    args = parser.parse_args()
    
    # Проверка существования директории
    if not os.path.isdir(args.audio_dir):
        print(f"❌ Директория не найдена: {args.audio_dir}")
        print("💡 Создайте директорию 'audio' или укажите путь через --audio-dir")
        sys.exit(1)
    
    try:
        batch_process(
            audio_dir=args.audio_dir,
            output_csv=args.output,
            c=args.speed_of_sound,
            block_start=args.block_start,
            block_len=args.block_len
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()