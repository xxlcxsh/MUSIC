#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D Near-field MUSIC Localization (Broadband ISSM)
Поддержка .aup / .wav, блочная обработка, CLI-параметры.
Исправлено: устойчивость к NaN/Inf, очистка данных, диагностика.
"""

import argparse
import os
import sys
import glob
import xml.etree.ElementTree as ET
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import soundfile as sf

# ================= КОНФИГУРАЦИЯ РЕШЁТКИ =================
NUM_MICS = 8
MIC_X = np.array([-0.14, -0.10, -0.06, -0.02, 0.02, 0.06, 0.10, 0.14])
MIC_Y = np.zeros(NUM_MICS)

# ================= ПАРАМЕТРЫ АЛГОРИТМА =================
FS_DEFAULT = 44100.0
FREQ_RANGE = (300.0, 4000.0)
THETA_DEG = np.arange(-90, 91, 1)
R_METERS = np.arange(0.2, 2.05, 0.05)
K_SOURCES = 1
# ========================================================

def strip_xml_namespaces(root):
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
    return root

def load_audacity_or_wav(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл не найден: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext in ['.wav', '.flac', '.ogg', '.aiff']:
        print(f"📂 Загрузка через soundfile: {filepath}")
        data, sr = sf.read(filepath, always_2d=True)
        X = data.T
        if X.shape[0] != NUM_MICS:
            raise ValueError(f"Ожидалось {NUM_MICS} каналов, получено {X.shape[0]}.")
        return X, sr

    if ext == '.aup':
        print(f"📂 Парсинг Audacity проекта: {filepath}")
        try:
            tree = ET.parse(filepath)
            root = strip_xml_namespaces(tree.getroot())
        except ET.ParseError as e:
            raise RuntimeError(f"Ошибка парсинга XML: {e}")

        project_dir = os.path.dirname(os.path.abspath(filepath))
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        data_dir = os.path.join(project_dir, f"{base_name}_data")
        if not os.path.isdir(data_dir):
            candidates = glob.glob(os.path.join(project_dir, "*_data"))
            data_dir = candidates[0] if candidates else data_dir

        tracks = root.findall('.//wavetrack')
        if len(tracks) < NUM_MICS:
            raise ValueError(f"В проекте {len(tracks)} дорожек. Требуется {NUM_MICS}.")

        X_list = []
        for idx, track in enumerate(tracks[:NUM_MICS]):
            blocks = track.findall('.//waveblock')
            if not blocks:
                raise ValueError(f"Дорожка {idx+1} пуста.")
            
            block_info = []
            for b in blocks:
                start = float(b.get('start', 0))
                fname = b.get('filename')
                if not fname:
                    sbf = b.find('.//simpleblockfile')
                    if sbf is not None:
                        fname = sbf.get('filename')
                if fname:
                    block_info.append((start, fname))
            
            block_info.sort(key=lambda x: x[0])
            track_samples = []
            for _, fname in block_info:
                au_path = os.path.join(data_dir, fname)
                if not os.path.exists(au_path):
                    found = glob.glob(os.path.join(data_dir, '**', fname), recursive=True)
                    au_path = found[0] if found else au_path
                track_samples.append(np.fromfile(au_path, dtype='<f4'))
            X_list.append(np.concatenate(track_samples))

        min_len = min(len(ch) for ch in X_list)
        X = np.array([ch[:min_len] for ch in X_list])
        print(f"✅ Собрано {X.shape[0]} каналов, {X.shape[1]} отсчётов.")
        return X, FS_DEFAULT

    if ext == '.aup3':
        raise NotImplementedError("Формат .aup3 не поддерживается. Экспортируйте в WAV.")
    raise ValueError(f"Неподдерживаемое расширение: {ext}")


def preprocess_signals(X, fs):
    print("🔧 Предобработка...")
    # 1. Центрирование
    X = X - np.mean(X, axis=1, keepdims=True)
    # 2. Фильтрация
    sos = sig.butter(4, FREQ_RANGE, btype='band', fs=fs, output='sos')
    X = sig.sosfilt(sos, X, axis=1)
    # 3. Очистка от выбросов и NaN/Inf (критично для STFT и eigh)
    X = np.clip(X, -1e3, 1e3)
    X = np.nan_to_num(X, nan=0.0, posinf=1e3, neginf=-1e3)
    
    if not np.all(np.isfinite(X)):
        raise ValueError("Сигнал содержит нечисловые значения после предобработки.")
    return X


def compute_nearfield_music(X, fs, c, mic_x, mic_y, thetas_deg, rs, freq_range):
    print("📐 Расчёт широкополосного MUSIC...")
    nperseg, noverlap = 1024, 512
    f, t, Zxx = sig.stft(X, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    # Безопасная очистка выхода STFT
    Zxx = np.nan_to_num(Zxx, nan=0.0, posinf=1e3, neginf=-1e3)
    
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    f_sel, Zxx_sel = f[freq_mask], Zxx[:, freq_mask, :]
    if len(f_sel) == 0:
        raise ValueError("Нет бинов в рабочем диапазоне частот.")
    print(f"   Рабочий диапазон: {freq_range[0]}-{freq_range[1]} Гц, бинов: {len(f_sel)}")
        
    thetas_rad = np.deg2rad(thetas_deg)
    R_grid, Theta_grid = np.meshgrid(rs, thetas_rad, indexing='ij')
    Xs = R_grid * np.sin(Theta_grid)
    Ys = R_grid * np.cos(Theta_grid)
    dists = np.sqrt((Xs[..., None] - mic_x)**2 + (Ys[..., None] - mic_y)**2)
    
    P_total = np.zeros_like(R_grid)
    valid_bins = 0
    skipped_eigh = 0
    
    for i, freq in enumerate(f_sel):
        X_f = Zxx_sel[:, i, :]
        T_frames = X_f.shape[1]
        if T_frames == 0:
            continue
            
        R_xx = (X_f @ X_f.conj().T) / T_frames
        # Принудительная эрмитовость и регуляризация
        R_xx = (R_xx + R_xx.conj().T) * 0.5
        R_xx += np.eye(NUM_MICS) * 1e-6 * (np.trace(R_xx) / NUM_MICS)
        
        try:
            w, v = np.linalg.eigh(R_xx)
        except np.linalg.LinAlgError:
            w, v = np.linalg.eig(R_xx)
            w = np.real(w)
            skipped_eigh += 1
            
        idx = np.argsort(w)[::-1]
        v = v[:, idx]
        En = v[:, K_SOURCES:]
        
        k = 2 * np.pi * freq / c
        A = np.exp(-1j * k * dists)
        EnH_A = np.einsum('ij,klj->kli', En.conj().T, A)
        denom = np.sum(np.abs(EnH_A)**2, axis=2)
        P_f = 1.0 / (denom + 1e-12)
        
        max_p = np.max(P_f)
        P_total += P_f / max_p
        valid_bins += 1
            
        if (i + 1) % 20 == 0 or i == len(f_sel) - 1:
            print(f"   Частотные бины: {i+1}/{len(f_sel)} (валидных: {valid_bins})")
            
    if valid_bins == 0:
        raise RuntimeError("Не удалось обработать ни один частотный бин. Проверьте данные или диапазон частот.")
    if skipped_eigh > 0:
        print(f"   ️ Пропущено разложений eigh: {skipped_eigh}")
        
    P_avg = P_total / valid_bins
    return 10 * np.log10(P_avg / np.max(P_avg) + 1e-12), P_avg


def plot_heatmap(P_dB, thetas_deg, rs, peak_theta, peak_r):
    print("🎨 Визуализация...")
    plt.figure(figsize=(10, 6))
    th_e = np.append(thetas_deg, thetas_deg[-1] + 1)
    r_e = np.append(rs, rs[-1] + 0.05)
    pcm = plt.pcolormesh(th_e, r_e, P_dB, shading='flat', cmap='inferno')
    plt.colorbar(pcm, label='MUSIC Spectrum (dB)').set_ticks([-40, -30, -20, -10, 0])
    plt.scatter(peak_theta, peak_r, c='cyan', marker='x', s=150, linewidths=3,
                label=f'Пик: θ={peak_theta:.1f}°, r={peak_r:.2f} м')
    plt.title('2D Near-field MUSIC (Broadband ISSM)', fontsize=14, pad=10)
    plt.xlabel('Угол θ (°)'); plt.ylabel('Расстояние r (м)')
    plt.grid(True, ls='--', alpha=0.5); plt.legend(loc='upper right')
    plt.tight_layout(); plt.show()


def run(audio_path, c=343.0, block_start=0, block_len=88200, show_plot=True):
    print(f"\n{'─'*60}")
    print(" 2D Near-field MUSIC Localization")
    print(f"{'─'*60}")
    
    X, sr = load_audacity_or_wav(audio_path)
    current_fs = sr
    if sr != FS_DEFAULT:
        print(f"⚠️ Fs файла ({sr} Гц) != конфиг ({FS_DEFAULT} Гц). Используется Fs файла.")

    end = min(block_start + block_len, X.shape[1])
    X_seg = X[:, block_start:end]
    if X_seg.shape[1] < 1024:
        raise ValueError("Блок слишком короткий для STFT (минимум 1024 отсчёта).")
    print(f"📦 Обрабатываем блок: [{block_start}:{end}] ({X_seg.shape[1]} отсчётов)")

    X_proc = preprocess_signals(X_seg, current_fs)
    P_dB, P_lin = compute_nearfield_music(X_proc, current_fs, c, MIC_X, MIC_Y, THETA_DEG, R_METERS, FREQ_RANGE)

    max_idx = np.unravel_index(np.argmax(P_lin), P_lin.shape)
    est_r = R_METERS[max_idx[0]]
    est_theta = THETA_DEG[max_idx[1]]

    print(f"\n{'═'*60}")
    print(" 🎯 РЕЗУЛЬТАТ ЛОКАЛИЗАЦИИ:")
    print(f"    Угол (градусы): {est_theta:.1f}")
    print(f"    Расстояние (м): {est_r:.2f}")
    print(f"{'═'*60}\n")

    if show_plot:
        plot_heatmap(P_dB, THETA_DEG, R_METERS, est_theta, est_r)
    return est_theta, est_r


def main():
    parser = argparse.ArgumentParser(description="2D Near-field MUSIC Sound Source Localization")
    parser.add_argument('audio_file', help='Путь к .aup или многоканальному .wav')
    parser.add_argument('--speed-of-sound', type=float, default=343.0, metavar='C', help='Скорость звука (м/с)')
    parser.add_argument('--block-start', type=int, default=0, help='Начальный отсчёт блока')
    parser.add_argument('--block-len', type=int, default=88200, help='Длина блока в отсчётах (по умолчанию 2 сек)')
    parser.add_argument('--no-plot', action='store_true', help='Отключить построение графика')
    args = parser.parse_args()

    try:
        run(
            audio_path=args.audio_file,
            c=args.speed_of_sound,
            block_start=args.block_start,
            block_len=args.block_len,
            show_plot=not args.no_plot
        )
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()