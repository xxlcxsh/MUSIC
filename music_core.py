#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Широкополосный near-field MUSIC по ТЗ (instruction.md)."""

from __future__ import annotations

import os
import sqlite3
import struct
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.signal as sig

try:
    import soundfile as sf

    HAS_SF = True
except ImportError:
    sf = None
    HAS_SF = False

# --- Геометрия: 8 микрофонов на окружности ---
NUM_MICS = 8
ARRAY_RADIUS_M = 0.5  # м, диаметр 1 м
_MIC_ANGLES = np.linspace(0.0, 2.0 * np.pi, NUM_MICS, endpoint=False)
MIC_X = (ARRAY_RADIUS_M * np.cos(_MIC_ANGLES)).astype(np.float64)
MIC_Y = (ARRAY_RADIUS_M * np.sin(_MIC_ANGLES)).astype(np.float64)
MIC_Z = np.zeros(NUM_MICS, dtype=np.float64)

# --- Параметры по умолчанию ---
FS_DEFAULT = 44100.0
C_SOUND_DEFAULT = 343.0
FREQ_RANGE_DEFAULT = (300.0, 4000.0)
K_SOURCES_DEFAULT = 1
D_MIN_DEFAULT = 0.2
D_MAX_DEFAULT = 3.0

STFT_NPERSEG = 1024
STFT_NOVERLAP = 512  # 50 %

_AU_DTYPE_MAP = {2: np.int8, 3: np.int16, 5: np.int32, 6: np.float32, 7: np.float64}


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


def _read_au_block(au_path: str) -> np.ndarray:
    """Читает Audacity .au блок как массив float32."""
    if HAS_SF:
        try:
            data, _ = sf.read(au_path, always_2d=False, dtype="float32")
            return np.asarray(data, dtype=np.float32)
        except Exception:
            pass

    with open(au_path, "rb") as handle:
        raw = handle.read()
    if len(raw) < 4:
        return np.zeros(0, dtype=np.float32)

    snd_magic = 0x2E736E64

    def _parse(endian: str) -> np.ndarray:
        magic, offset, _, encoding, _, _ = struct.unpack_from(f"{endian}IIIIII", raw)
        if magic != snd_magic or encoding not in _AU_DTYPE_MAP:
            raise ValueError
        dtype = np.dtype(_AU_DTYPE_MAP[encoding]).newbyteorder(endian)
        samples = np.frombuffer(raw, dtype=dtype, offset=offset)
        if samples.dtype.kind in ("i", "u"):
            samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
        return samples.astype(np.float32)

    for endian in (">", "<"):
        try:
            return _parse(endian)
        except Exception:
            pass

    warnings.warn(f"Fallback raw int16: {os.path.basename(au_path)}")
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _load_track(block_files: list[str]) -> np.ndarray:
    return np.concatenate([_read_au_block(path) for path in block_files])


def load_aup_xml(aup_path: str):
    """Загрузка Audacity .aup проекта и связанных .au блоков."""
    tree = ET.parse(aup_path)
    root = tree.getroot()
    rate = float(root.get("rate", 44100))
    base_dir = os.path.dirname(os.path.abspath(aup_path))
    uri = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""

    def tag(name: str) -> str:
        return f"{{{uri}}}{name}" if uri else name

    tracks: list[np.ndarray] = []
    for wavetrack in root.iter(tag("wavetrack")):
        blocks: dict[int, str] = {}
        for seq in wavetrack.iter(tag("sequence")):
            for blk in seq.iter(tag("simpleblockfile")):
                fname = blk.get("filename")
                offset = int(blk.get("start", 0))
                if fname:
                    for dirpath, _, files in os.walk(base_dir):
                        if fname in files:
                            blocks[offset] = os.path.join(dirpath, fname)
                            break
        if blocks:
            tracks.append(_load_track([blocks[k] for k in sorted(blocks)]))

    if not tracks:
        raise RuntimeError("Треки не найдены. Убедитесь что папка _data находится рядом с .aup файлом.")
    return tracks, rate


def load_aup3_sqlite(path: str):
    """Загрузка Audacity .aup3 проекта через SQLite."""
    con = sqlite3.connect(path)
    cur = con.cursor()
    try:
        cur.execute("SELECT dict FROM project LIMIT 1")
        import json

        rate = float(json.loads(cur.fetchone()[0]).get("rate", 44100))
    except Exception:
        rate = 44100.0
    cur.execute("SELECT DISTINCT trackid FROM sampleblocks ORDER BY trackid")
    tracks: list[np.ndarray] = []
    for (tid,) in cur.fetchall():
        cur.execute("SELECT samples FROM sampleblocks WHERE trackid=? ORDER BY blockid", (tid,))
        parts = [np.frombuffer(blob, dtype="<f4") for (blob,) in cur.fetchall()]
        if parts:
            tracks.append(np.concatenate(parts))
    con.close()
    if not tracks:
        raise RuntimeError("Нет блоков в .aup3 файле")
    return tracks, rate


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
        tracks, sr = load_aup_xml(filepath)
        if len(tracks) < NUM_MICS:
            raise ValueError(f"В проекте {len(tracks)} дорожек, требуется {NUM_MICS}.")
        min_len = min(len(t) for t in tracks[:NUM_MICS])
        x = np.array([tracks[i][:min_len] for i in range(NUM_MICS)], dtype=np.float64)
        return x, float(sr)

    if ext == ".aup3":
        tracks, sr = load_aup3_sqlite(filepath)
        if len(tracks) < NUM_MICS:
            raise ValueError(f"В проекте {len(tracks)} дорожек, требуется {NUM_MICS}.")
        min_len = min(len(t) for t in tracks[:NUM_MICS])
        x = np.array([tracks[i][:min_len] for i in range(NUM_MICS)], dtype=np.float64)
        return x, float(sr)
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


def _grid_positions_circular_planar(
    azimuth_deg: np.ndarray,
    d_m: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Источник в плоскости XY; азимут в системе 0°..360°:
    0° = ось +X, против часовой стрелки; x = d·cos(θ), y = d·sin(θ).
    """
    tt, dd = np.meshgrid(np.deg2rad(azimuth_deg), d_m, indexing="ij")
    sx = dd * np.cos(tt)
    sy = dd * np.sin(tt)
    sz = np.zeros_like(sx)
    return sx.ravel(), sy.ravel(), sz.ravel(), dd.ravel()


def _music_quadratic_denom(steering: np.ndarray, p_n: np.ndarray) -> np.ndarray:
    """a^H P_n a для пакета векторов управления (G, M)."""
    sp = steering @ p_n
    return np.real(np.sum(steering * np.conj(sp), axis=1))


def _music_spectrum_farfield_doa(
    azimuth_deg: np.ndarray,
    mic_x: np.ndarray,
    mic_y: np.ndarray,
    projectors: list[np.ndarray],
    freqs: np.ndarray,
    c: float,
) -> np.ndarray:
    """Широкополосный MUSIC (дальнее поле) по азимуту 0–360°."""
    theta = np.deg2rad(azimuth_deg)
    cos_t = np.cos(theta)[:, None]
    sin_t = np.sin(theta)[:, None]
    p_sum = np.zeros(len(azimuth_deg), dtype=np.float64)

    for p_n, freq in zip(projectors, freqs):
        tau = -(mic_x[None, :] * cos_t + mic_y[None, :] * sin_t) / c
        steering = np.exp(-1j * 2.0 * np.pi * freq * tau)
        p_sum += 1.0 / (_music_quadratic_denom(steering, p_n) + 1e-12)

    return p_sum


def _fine_azimuth_range(
    azimuth_peak: float,
    half_width: float,
    step: float,
) -> np.ndarray:
    """Узкий диапазон азимута с учётом перехода через 0°/360°."""
    center = azimuth_peak % 360.0
    n = int(np.ceil(2.0 * half_width / step)) + 1
    offsets = np.linspace(-half_width, half_width, n)
    return (center + offsets) % 360.0


def legacy_azimuth_to_instruction(azimuth_deg: float) -> float:
    """Азимут уже в системе instruction.md (0–360°, от +X)."""
    return float(azimuth_deg % 360.0)


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
        p_sum += 1.0 / (_music_quadratic_denom(steering, p_n) + 1e-12)

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

    planar_mode=True (по умолчанию): круговая решётка, MUSIC-поиск по азимуту
    и расстоянию в плоскости XY с coarse-to-fine сеткой.
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
        # --- Планарный MUSIC: азимут + расстояние на круговой решётке ---
        az_c = np.arange(0.0, 360.0, 5.0)
        d_c = np.arange(d_min, d_max + 1e-9, 0.2)
        sx, sy, sz, d_flat = _grid_positions_circular_planar(az_c, d_c)
        spec_coarse = _music_spectrum_on_grid(
            sx, sy, sz, d_flat, mic_x, mic_y, mic_z, projectors, freqs_used, c
        )
        i_az, i_d, _ = _argmax_planar(spec_coarse, az_c, d_c)
        az_peak = float(az_c[i_az])
        d_peak = float(d_c[i_d])

        az_f = _fine_azimuth_range(az_peak, 10.0, 1.0)
        d_f = np.clip(np.arange(d_peak - 0.2, d_peak + 0.2 + 0.005, 0.01), d_min, d_max)
        sx, sy, sz, d_flat = _grid_positions_circular_planar(az_f, d_f)
        spec_fine = _music_spectrum_on_grid(
            sx, sy, sz, d_flat, mic_x, mic_y, mic_z, projectors, freqs_used, c
        )
        j_az, j_d, peak_val = _argmax_planar(spec_fine, az_f, d_f)

        result = MusicResult(
            azimuth_deg=float(az_f[j_az]),
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
