#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Широкополосный near-field MUSIC по ТЗ (instruction.md)."""

from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.signal as sig

# --- Геометрия решётки (из music.py) ---
NUM_MICS = 8
MIC_X = np.array([-0.14, -0.10, -0.06, -0.02, 0.02, 0.06, 0.10, 0.14], dtype=np.float64)
MIC_Y = np.zeros(NUM_MICS, dtype=np.float64)
MIC_Z = np.zeros(NUM_MICS, dtype=np.float64)
ARRAY_RADIUS_M = float(np.max(np.hypot(MIC_X, MIC_Y)))

# --- Параметры по умолчанию ---
FS_DEFAULT = 44100.0
C_SOUND_DEFAULT = 343.0
FREQ_RANGE_DEFAULT = (300.0, 4000.0)
K_SOURCES_DEFAULT = 1
D_MIN_DEFAULT = 0.2
D_MAX_DEFAULT = 3.0

STFT_NPERSEG = 1024
STFT_NOVERLAP = 512  # 50 %


@dataclass
class MusicResult:
    azimuth_deg: float
    elevation_deg: float
    distance_m: float
    spectrum_coarse: Optional[np.ndarray] = None
    spectrum_fine: Optional[np.ndarray] = None
    grid_coarse: Optional[dict] = None
    grid_fine: Optional[dict] = None
    frequencies_hz: Optional[np.ndarray] = None
    eigenvalues: Optional[np.ndarray] = None
    stft: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None


def _exchange_matrix(m: int) -> np.ndarray:
    return np.fliplr(np.eye(m, dtype=np.float64))


_J8 = _exchange_matrix(NUM_MICS)


def read_audacity_au_block(au_path: str, num_samples: Optional[int] = None) -> np.ndarray:
    """
    Чтение блока Audacity 2.x (заголовок dns. + 32-бит float PCM).
    """
    with open(au_path, "rb") as handle:
        magic = handle.read(4)
        if magic != b"dns.":
            raise ValueError(f"Неизвестный формат AU-блока: {au_path}")
        handle.seek(24)
        if num_samples is None:
            payload = handle.read()
        else:
            payload = handle.read(int(num_samples) * 4)
    samples = np.frombuffer(payload, dtype="<f4").copy()
    if num_samples is not None and samples.size > num_samples:
        samples = samples[:num_samples]
    bad = ~np.isfinite(samples) | (np.abs(samples) > 10.0)
    if np.any(bad):
        samples[bad] = 0.0
    return samples


def strip_xml_namespaces(root: ET.Element) -> ET.Element:
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]
    return root


def load_audacity_or_wav(filepath: str) -> Tuple[np.ndarray, float]:
    """Загрузка 8-канального .wav/.flac или проекта Audacity .aup."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл не найден: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext in (".wav", ".flac", ".ogg", ".aiff"):
        import soundfile as sf

        data, sr = sf.read(filepath, always_2d=True)
        x = data.T
        if x.shape[0] != NUM_MICS:
            raise ValueError(f"Ожидалось {NUM_MICS} каналов, получено {x.shape[0]}.")
        return x.astype(np.float64), float(sr)

    if ext == ".aup":
        tree = ET.parse(filepath)
        root = strip_xml_namespaces(tree.getroot())
        project_dir = os.path.dirname(os.path.abspath(filepath))
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        data_dir = os.path.join(project_dir, f"{base_name}_data")
        if not os.path.isdir(data_dir):
            candidates = glob.glob(os.path.join(project_dir, "*_data"))
            data_dir = candidates[0] if candidates else data_dir

        tracks = root.findall(".//wavetrack")
        if len(tracks) < NUM_MICS:
            raise ValueError(f"В проекте {len(tracks)} дорожек, требуется {NUM_MICS}.")

        channels = []
        for track in tracks[:NUM_MICS]:
            blocks = track.findall(".//waveblock")
            if not blocks:
                raise ValueError("Пустая дорожка в проекте.")
            block_info = []
            for block in blocks:
                start = float(block.get("start", 0))
                sbf = block.find(".//simpleblockfile")
                fname = block.get("filename")
                length = None
                if sbf is not None:
                    fname = fname or sbf.get("filename")
                    if sbf.get("len"):
                        length = int(sbf.get("len"))
                if fname:
                    block_info.append((start, fname, length))
            block_info.sort(key=lambda x: x[0])
            samples = []
            for _, fname, length in block_info:
                au_path = os.path.join(data_dir, fname)
                if not os.path.exists(au_path):
                    found = glob.glob(os.path.join(data_dir, "**", fname), recursive=True)
                    au_path = found[0] if found else au_path
                samples.append(read_audacity_au_block(au_path, length))
            channels.append(np.concatenate(samples))

        min_len = min(len(ch) for ch in channels)
        x = np.array([ch[:min_len] for ch in channels], dtype=np.float64)
        return x, FS_DEFAULT

    if ext == ".aup3":
        raise NotImplementedError("Формат .aup3 не поддерживается. Экспортируйте в WAV.")
    raise ValueError(f"Неподдерживаемое расширение: {ext}")


def preprocess_signals(x: np.ndarray, fs: float, freq_range: Tuple[float, float]) -> np.ndarray:
    """Центрирование, полосовая фильтрация, очистка выбросов."""
    x = x - np.mean(x, axis=1, keepdims=True)
    sos = sig.butter(4, freq_range, btype="band", fs=fs, output="sos")
    x = sig.sosfilt(sos, x, axis=1)
    x = np.clip(x, -1e3, 1e3)
    x = np.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
    if not np.all(np.isfinite(x)):
        raise ValueError("Сигнал содержит нечисловые значения после предобработки.")
    return x


def _compute_stft(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f, t, zxx = sig.stft(
        x,
        fs=fs,
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
        window="hann",
        boundary=None,
    )
    zxx = np.nan_to_num(zxx, nan=0.0, posinf=0.0, neginf=0.0)
    return f, t, zxx


def _forward_backward_cov(r: np.ndarray) -> np.ndarray:
    return 0.5 * (r + _J8 @ r.conj() @ _J8)


def _noise_projectors(zxx_f: np.ndarray, num_sources: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    zxx_f: (M, T_frames) для одной частоты.
    Возвращает (P_n, eigenvalues_sorted_desc).
    """
    t_frames = zxx_f.shape[1]
    if t_frames < 2:
        return None, None

    r = (zxx_f @ zxx_f.conj().T) / t_frames
    r = _forward_backward_cov(r)
    r = 0.5 * (r + r.conj().T)
    r += np.eye(NUM_MICS) * 1e-8 * (np.trace(r).real / NUM_MICS + 1e-12)

    w, v = np.linalg.eigh(r)
    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]
    u_n = v[:, num_sources:]
    p_n = u_n @ u_n.conj().T
    return p_n, w


def _grid_positions(
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    d_m: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Сферическая сетка по формулам из instruction.md."""
    tt, pp, dd = np.meshgrid(
        np.deg2rad(theta_deg),
        np.deg2rad(phi_deg),
        d_m,
        indexing="ij",
    )
    sx = dd * np.cos(tt) * np.sin(pp)
    sy = dd * np.sin(tt) * np.sin(pp)
    sz = dd * np.cos(pp)
    return sx.ravel(), sy.ravel(), sz.ravel(), dd.ravel()


def _grid_positions_planar(
    azimuth_deg: np.ndarray,
    d_m: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Планарная модель для линейной решётки (координаты из music.py):
    x = d·sin(θ), y = d·cos(θ), z = 0.
    Азимут θ — угол от оси +Y, совпадает с разметкой в именах файлов.
    """
    tt, dd = np.meshgrid(np.deg2rad(azimuth_deg), d_m, indexing="ij")
    sx = dd * np.sin(tt)
    sy = dd * np.cos(tt)
    sz = np.zeros_like(sx)
    return sx.ravel(), sy.ravel(), sz.ravel(), dd.ravel()


def legacy_azimuth_to_instruction(azimuth_legacy_deg: float) -> float:
    """Преобразование в азимут instruction.md (0–360°, от оси +X в плоскости XY)."""
    rad = np.deg2rad(azimuth_legacy_deg)
    return float(np.degrees(np.arctan2(np.cos(rad), np.sin(rad))) % 360.0)


def _music_spectrum_on_grid(
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    d_flat: np.ndarray,
    mic_x: np.ndarray,
    mic_y: np.ndarray,
    mic_z: np.ndarray,
    projectors: list[np.ndarray],
    freqs: np.ndarray,
    c: float,
) -> np.ndarray:
    """Векторизованный широкополосный MUSIC: сумма 1/||P_n a||^2 по частотам."""
    g = sx.shape[0]
    p_sum = np.zeros(g, dtype=np.float64)

    diff_x = sx[:, None] - mic_x[None, :]
    diff_y = sy[:, None] - mic_y[None, :]
    diff_z = sz[:, None] - mic_z[None, :]
    d_mics = np.sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z)

    for p_n, freq in zip(projectors, freqs):
        phase = np.exp(-1j * 2.0 * np.pi * freq / c * (d_mics - d_flat[:, None]))
        steering = (d_flat[:, None] / np.maximum(d_mics, 1e-9)) * phase
        proj = steering @ p_n.T
        denom = np.sum(np.abs(proj) ** 2, axis=1)
        p_sum += 1.0 / (denom + 1e-12)

    return p_sum


def _argmax_on_grid(
    spectrum: np.ndarray,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    d_m: np.ndarray,
) -> Tuple[int, int, int, float]:
    n_theta, n_phi, n_d = len(theta_deg), len(phi_deg), len(d_m)
    spec_3d = spectrum.reshape(n_theta, n_phi, n_d)
    idx = np.unravel_index(np.argmax(spec_3d), spec_3d.shape)
    return idx[0], idx[1], idx[2], float(spec_3d[idx])


def _argmax_planar(
    spectrum: np.ndarray,
    azimuth_deg: np.ndarray,
    d_m: np.ndarray,
) -> Tuple[int, int, float]:
    spec_2d = spectrum.reshape(len(azimuth_deg), len(d_m))
    idx = np.unravel_index(np.argmax(spec_2d), spec_2d.shape)
    return idx[0], idx[1], float(spec_2d[idx])


def _fine_ranges(
    theta_c: float,
    phi_c: float,
    d_c: float,
    theta_step: float,
    phi_step: float,
    d_step: float,
    d_min: float,
    d_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.arange(theta_c - 10.0, theta_c + 10.0 + 0.5 * theta_step, theta_step) % 360.0
    phi = np.clip(np.arange(phi_c - 10.0, phi_c + 10.0 + 0.5 * phi_step, phi_step), 0.0, 90.0)
    d = np.clip(np.arange(d_c - 0.2, d_c + 0.2 + 0.5 * d_step, d_step), d_min, d_max)
    return theta, phi, d


def music_localization(
    audio_data: np.ndarray,
    fs: float,
    num_sources: int = K_SOURCES_DEFAULT,
    freq_range: Tuple[float, float] = FREQ_RANGE_DEFAULT,
    c: float = C_SOUND_DEFAULT,
    d_min: float = D_MIN_DEFAULT,
    d_max: float = D_MAX_DEFAULT,
    mic_x: np.ndarray = MIC_X,
    mic_y: np.ndarray = MIC_Y,
    mic_z: np.ndarray = MIC_Z,
    return_diagnostics: bool = False,
    planar_mode: bool = True,
    full_3d: bool = False,
) -> MusicResult:
    """
    Локализация источника (coarse-to-fine, векторизованный MUSIC).

    planar_mode=True (по умолчанию): 2D near-field для линейной решётки music.py,
    азимут в legacy-градусах (от +Y), elevation=90°.
    full_3d=True: полная сетка (θ, φ, d) по instruction.md.
    """
    if audio_data.shape[0] != NUM_MICS:
        raise ValueError(f"Ожидалось {NUM_MICS} каналов, получено {audio_data.shape[0]}.")
    if audio_data.shape[1] < STFT_NPERSEG:
        raise ValueError(f"Нужно минимум {STFT_NPERSEG} отсчётов.")

    x = preprocess_signals(audio_data, fs, freq_range)
    f_hz, t_stft, zxx = _compute_stft(x, fs)

    mask = (f_hz >= freq_range[0]) & (f_hz <= freq_range[1])
    f_sel = f_hz[mask]
    z_sel = zxx[:, mask, :]
    if len(f_sel) == 0:
        raise ValueError("Нет частотных бинов в рабочем диапазоне.")

    projectors: list[np.ndarray] = []
    all_eigs: list[np.ndarray] = []
    for i in range(len(f_sel)):
        p_n, w = _noise_projectors(z_sel[:, i, :], num_sources)
        if p_n is None:
            continue
        projectors.append(p_n)
        all_eigs.append(w)

    if not projectors:
        raise RuntimeError("Не удалось построить матрицы проекции ни для одной частоты.")

    freqs_used = f_sel[: len(projectors)]

    use_planar = planar_mode and not full_3d

    if use_planar:
        # --- Planar coarse (θ: -90..90, d) ---
        az_c = np.arange(-90.0, 90.0 + 1e-9, 5.0)
        d_c = np.arange(d_min, d_max + 1e-9, 0.2)
        sx, sy, sz, d_flat = _grid_positions_planar(az_c, d_c)
        spec_coarse = _music_spectrum_on_grid(
            sx, sy, sz, d_flat, mic_x, mic_y, mic_z, projectors, freqs_used, c
        )
        i_a, i_d, _ = _argmax_planar(spec_coarse, az_c, d_c)
        az_peak = float(az_c[i_a])
        d_peak = float(d_c[i_d])

        az_f = np.clip(np.arange(az_peak - 10.0, az_peak + 10.0 + 0.5, 1.0), -90.0, 90.0)
        d_f = np.clip(np.arange(d_peak - 0.2, d_peak + 0.2 + 0.005, 0.01), d_min, d_max)
        sx, sy, sz, d_flat = _grid_positions_planar(az_f, d_f)
        spec_fine = _music_spectrum_on_grid(
            sx, sy, sz, d_flat, mic_x, mic_y, mic_z, projectors, freqs_used, c
        )
        j_a, j_d, peak_val = _argmax_planar(spec_fine, az_f, d_f)

        az_legacy = float(az_f[j_a])
        result = MusicResult(
            azimuth_deg=az_legacy,
            elevation_deg=90.0,
            distance_m=float(d_f[j_d]),
        )
        if return_diagnostics:
            result.spectrum_coarse = spec_coarse.reshape(len(az_c), len(d_c))
            result.spectrum_fine = spec_fine.reshape(len(az_f), len(d_f))
            result.grid_coarse = {"theta": az_c, "phi": np.array([90.0]), "d": d_c, "planar": True}
            result.grid_fine = {"theta": az_f, "phi": np.array([90.0]), "d": d_f, "planar": True}
    else:
        # --- Full 3D coarse-to-fine (instruction.md) ---
        theta_c = np.arange(0.0, 360.0, 5.0)
        phi_c = np.arange(0.0, 90.0 + 1e-9, 5.0)
        d_c = np.arange(d_min, d_max + 1e-9, 0.2)

        sx, sy, sz, d_flat = _grid_positions(theta_c, phi_c, d_c)
        spec_coarse = _music_spectrum_on_grid(
            sx, sy, sz, d_flat, mic_x, mic_y, mic_z, projectors, freqs_used, c
        )
        i_t, i_p, i_d, _ = _argmax_on_grid(spec_coarse, theta_c, phi_c, d_c)
        theta_peak = float(theta_c[i_t])
        phi_peak = float(phi_c[i_p])
        d_peak = float(d_c[i_d])

        theta_f, phi_f, d_f = _fine_ranges(
            theta_peak, phi_peak, d_peak, 1.0, 1.0, 0.01, d_min, d_max
        )
        sx, sy, sz, d_flat = _grid_positions(theta_f, phi_f, d_f)
        spec_fine = _music_spectrum_on_grid(
            sx, sy, sz, d_flat, mic_x, mic_y, mic_z, projectors, freqs_used, c
        )
        j_t, j_p, j_d, peak_val = _argmax_on_grid(spec_fine, theta_f, phi_f, d_f)

        result = MusicResult(
            azimuth_deg=float(theta_f[j_t]),
            elevation_deg=float(phi_f[j_p]),
            distance_m=float(d_f[j_d]),
        )
        if return_diagnostics:
            result.spectrum_coarse = spec_coarse.reshape(len(theta_c), len(phi_c), len(d_c))
            result.spectrum_fine = spec_fine.reshape(len(theta_f), len(phi_f), len(d_f))
            result.grid_coarse = {"theta": theta_c, "phi": phi_c, "d": d_c, "planar": False}
            result.grid_fine = {"theta": theta_f, "phi": phi_f, "d": d_f, "planar": False}

    if return_diagnostics:
        result.frequencies_hz = freqs_used
        result.eigenvalues = np.stack(all_eigs, axis=0) if all_eigs else None
        result.stft = (f_hz, t_stft, zxx)
        _ = peak_val

    return result


def music_octagon_localization(
    audio_data: np.ndarray,
    fs: float,
    r: float = ARRAY_RADIUS_M,
    num_sources: int = K_SOURCES_DEFAULT,
    freq_range: Tuple[float, float] = FREQ_RANGE_DEFAULT,
) -> Tuple[float, float, float]:
    """Интерфейс из ТЗ; r сохранён для совместимости, координаты — из music.py."""
    _ = r
    res = music_localization(
        audio_data,
        fs,
        num_sources=num_sources,
        freq_range=freq_range,
        planar_mode=True,
        full_3d=False,
    )
    az_instr = legacy_azimuth_to_instruction(res.azimuth_deg)
    return az_instr, res.elevation_deg, res.distance_m


def parse_ground_truth_from_filename(path: str) -> Optional[Tuple[float, float]]:
    """
    A1_CH_10_20 -> азимут 10°, дистанция 0.20 м (последнее число / 100).
    """
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("_")
    if len(parts) < 4:
        return None
    try:
        azimuth = float(parts[-2])
        distance = float(parts[-1]) / 100.0
        return azimuth, distance
    except ValueError:
        return None
