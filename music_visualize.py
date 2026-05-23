#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Подробные визуализации работы алгоритма MUSIC."""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from music_core import MIC_X, MIC_Y, MIC_Z, MusicResult, NUM_MICS


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_all_visualizations(
    result: MusicResult,
    out_dir: str,
    title_prefix: str = "",
    ground_truth: Optional[tuple[float, float]] = None,
) -> None:
    """Сохраняет полный набор диагностических графиков в out_dir."""
    out = _ensure_dir(out_dir)
    prefix = f"{title_prefix}_" if title_prefix else ""

    if result.stft is not None:
        _plot_stft_channels(result, os.path.join(out, f"{prefix}01_stft_channels.png"))

    if result.eigenvalues is not None and result.frequencies_hz is not None:
        _plot_eigenvalues(result, os.path.join(out, f"{prefix}02_eigenvalues.png"))

    if result.spectrum_coarse is not None and result.grid_coarse is not None:
        _plot_spectrum_slices(
            result.spectrum_coarse,
            result.grid_coarse,
            result.azimuth_deg,
            result.elevation_deg,
            result.distance_m,
            "Coarse",
            out,
            prefix,
            "03",
            ground_truth,
        )

    if result.spectrum_fine is not None and result.grid_fine is not None:
        _plot_spectrum_slices(
            result.spectrum_fine,
            result.grid_fine,
            result.azimuth_deg,
            result.elevation_deg,
            result.distance_m,
            "Fine",
            out,
            prefix,
            "04",
            ground_truth,
        )

    _plot_geometry_3d(result, os.path.join(out, f"{prefix}05_geometry_3d.png"), ground_truth)
    _plot_polar_azimuth(result, os.path.join(out, f"{prefix}06_polar_azimuth.png"))
    _plot_summary_card(result, os.path.join(out, f"{prefix}07_summary.png"), ground_truth, title_prefix)


def _plot_stft_channels(result: MusicResult, path: str) -> None:
    f_hz, t_s, zxx = result.stft
    fig, axes = plt.subplots(NUM_MICS, 1, figsize=(12, 2.2 * NUM_MICS), sharex=True)
    if NUM_MICS == 1:
        axes = [axes]
    for m in range(NUM_MICS):
        mag = 20 * np.log10(np.abs(zxx[m]) + 1e-12)
        im = axes[m].pcolormesh(t_s, f_hz, mag, shading="auto", cmap="magma")
        axes[m].set_ylabel(f"Ch{m + 1} (dB)")
        axes[m].set_ylim(0, min(8000, f_hz[-1]))
    axes[-1].set_xlabel("Время, с")
    fig.colorbar(im, ax=axes, label="|STFT| dB", fraction=0.02, pad=0.02)
    fig.suptitle("STFT по каналам (окно Hann, overlap 50%)", fontsize=13)
    _savefig(path)


def _plot_eigenvalues(result: MusicResult, path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    eigs = result.eigenvalues
    freqs = result.frequencies_hz
    for k in range(eigs.shape[1]):
        ax.plot(freqs, eigs[:, k], label=f"λ{k + 1}", linewidth=1.2)
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("Собственное значение R_fb")
    ax.set_title("Спектр собственных значений ковариационной матрицы (после FB-усреднения)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.35)
    ax.legend(ncol=4, fontsize=8)
    _savefig(path)


def _plot_spectrum_slices(
    spectrum: np.ndarray,
    grid: dict,
    az: float,
    el: float,
    dist: float,
    stage: str,
    out_dir: str,
    prefix: str,
    idx: str,
    ground_truth: Optional[tuple[float, float]],
) -> None:
    theta = grid["theta"]
    phi = grid["phi"]
    d = grid["d"]
    spec_db = 10 * np.log10(spectrum / (spectrum.max() + 1e-12) + 1e-12)

    if grid.get("planar"):
        if spec_db.ndim == 1:
            _plot_srp_azimuth_spectrum(spec_db, theta, az, stage, out_dir, prefix, idx)
        else:
            _plot_planar_spectrum(spec_db, theta, d, az, dist, stage, out_dir, prefix, idx, ground_truth)
        return

    i_phi = int(np.argmin(np.abs(phi - el)))
    i_d = int(np.argmin(np.abs(d - dist)))
    i_theta = int(np.argmin(np.abs(theta - az)))

    # θ–d при фиксированном φ
    fig, ax = plt.subplots(figsize=(10, 5))
    pcm = ax.pcolormesh(theta, d, spec_db[:, i_phi, :].T, shading="auto", cmap="inferno")
    ax.scatter([az], [dist], c="cyan", marker="x", s=120, linewidths=2, label="Оценка")
    if ground_truth:
        ax.scatter([ground_truth[0]], [ground_truth[1]], c="lime", marker="+", s=120, linewidths=2, label="Эталон (имя файла)")
    ax.set_xlabel("Азимут θ, °")
    ax.set_ylabel("Расстояние d, м")
    ax.set_title(f"{stage}: MUSIC (θ–d), φ={phi[i_phi]:.0f}°")
    fig.colorbar(pcm, ax=ax, label="dB отн. макс.")
    ax.legend()
    _savefig(os.path.join(out_dir, f"{prefix}{idx}a_{stage.lower()}_theta_distance.png"))

    # θ–φ при фиксированном d
    fig, ax = plt.subplots(figsize=(10, 5))
    pcm = ax.pcolormesh(theta, phi, spec_db[:, :, i_d], shading="auto", cmap="inferno")
    ax.scatter([az], [el], c="cyan", marker="x", s=120, linewidths=2, label="Оценка")
    ax.set_xlabel("Азимут θ, °")
    ax.set_ylabel("Угол места φ, °")
    ax.set_title(f"{stage}: MUSIC (θ–φ), d={d[i_d]:.2f} м")
    fig.colorbar(pcm, ax=ax, label="dB отн. макс.")
    ax.legend()
    _savefig(os.path.join(out_dir, f"{prefix}{idx}b_{stage.lower()}_theta_elevation.png"))

    # φ–d при фиксированном θ
    fig, ax = plt.subplots(figsize=(10, 5))
    pcm = ax.pcolormesh(phi, d, spec_db[i_theta, :, :].T, shading="auto", cmap="inferno")
    ax.scatter([el], [dist], c="cyan", marker="x", s=120, linewidths=2, label="Оценка")
    ax.set_xlabel("Угол места φ, °")
    ax.set_ylabel("Расстояние d, м")
    ax.set_title(f"{stage}: MUSIC (φ–d), θ={theta[i_theta]:.0f}°")
    fig.colorbar(pcm, ax=ax, label="dB отн. макс.")
    ax.legend()
    _savefig(os.path.join(out_dir, f"{prefix}{idx}c_{stage.lower()}_elevation_distance.png"))

    # 1D срезы через пик
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(theta, spec_db[:, i_phi, i_d])
    axes[0].axvline(az, color="cyan", ls="--", label="пик")
    axes[0].set_xlabel("θ, °")
    axes[0].set_ylabel("dB")
    axes[0].set_title("Срез по θ")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(phi, spec_db[i_theta, :, i_d])
    axes[1].axvline(el, color="cyan", ls="--")
    axes[1].set_xlabel("φ, °")
    axes[1].set_title("Срез по φ")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(d, spec_db[i_theta, i_phi, :])
    axes[2].axvline(dist, color="cyan", ls="--")
    if ground_truth:
        axes[2].axvline(ground_truth[1], color="lime", ls=":", label="эталон d")
    axes[2].set_xlabel("d, м")
    axes[2].set_title("Срез по d")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)
    fig.suptitle(f"{stage}: одномерные срезы через оценку", fontsize=12)
    _savefig(os.path.join(out_dir, f"{prefix}{idx}d_{stage.lower()}_1d_slices.png"))


def _plot_geometry_3d(result: MusicResult, path: str, ground_truth: Optional[tuple[float, float]]) -> None:
    d = result.distance_m
    az = np.deg2rad(result.azimuth_deg)
    sx = d * np.cos(az)
    sy = d * np.sin(az)
    sz = 0.0

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(MIC_X, MIC_Y, MIC_Z, c="tab:blue", s=80, label="Микрофоны")
    for i in range(NUM_MICS):
        ax.text(MIC_X[i], MIC_Y[i], MIC_Z[i], f" {i + 1}", fontsize=8)

    ax.scatter([0], [0], [0], c="black", marker="o", s=40, label="Центр решётки")
    ax.scatter([sx], [sy], [sz], c="red", marker="*", s=200, label="Оценка источника")

    if ground_truth:
        gt_az = np.deg2rad(ground_truth[0])
        gt_d = ground_truth[1]
        gx = gt_d * np.cos(gt_az)
        gy = gt_d * np.sin(gt_az)
        ax.scatter([gx], [gy], [0], c="lime", marker="^", s=120, label="Эталон (legacy)")

    ax.plot([0, sx], [0, sy], [0, sz], "r--", alpha=0.5)
    ax.set_xlabel("X, м")
    ax.set_ylabel("Y, м")
    ax.set_zlabel("Z, м")
    ax.set_title("Геометрия: решётка и оценённое положение источника")
    ax.legend(loc="upper left", fontsize=8)
    lim = max(0.2, d * 1.2, 0.16)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-0.05, lim)
    _savefig(path)


def _plot_srp_azimuth_spectrum(
    spec_db: np.ndarray,
    azimuth_deg: np.ndarray,
    az: float,
    stage: str,
    out_dir: str,
    prefix: str,
    idx: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(azimuth_deg, spec_db, color="tab:blue")
    ax.axvline(az, color="cyan", ls="--", label=f"пик {az:.1f}°")
    ax.set_xlabel("Азимут (0° = +X, ПЧС), °")
    ax.set_ylabel("dB отн. макс.")
    ax.set_title(f"{stage}: MUSIC по азимуту")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _savefig(os.path.join(out_dir, f"{prefix}{idx}_srp_azimuth.png"))


def _plot_planar_spectrum(
    spec_db: np.ndarray,
    azimuth_deg: np.ndarray,
    d_m: np.ndarray,
    az: float,
    dist: float,
    stage: str,
    out_dir: str,
    prefix: str,
    idx: str,
    ground_truth: Optional[tuple[float, float]],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    pcm = ax.pcolormesh(azimuth_deg, d_m, spec_db.T, shading="auto", cmap="inferno")
    ax.scatter([az], [dist], c="cyan", marker="x", s=120, linewidths=2, label="Оценка")
    if ground_truth:
        ax.scatter([ground_truth[0]], [ground_truth[1]], c="lime", marker="+", s=120, linewidths=2, label="Эталон")
    ax.set_xlabel("Азимут (0° = +X, ПЧС), °")
    ax.set_ylabel("Расстояние d, м")
    ax.set_title(f"{stage}: планарный MUSIC (азимут–расстояние)")
    fig.colorbar(pcm, ax=ax, label="dB отн. макс.")
    ax.legend()
    _savefig(os.path.join(out_dir, f"{prefix}{idx}_planar_azimuth_distance.png"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    i_d = int(np.argmin(np.abs(d_m - dist)))
    i_a = int(np.argmin(np.abs(azimuth_deg - az)))
    axes[0].plot(azimuth_deg, spec_db[:, i_d])
    axes[0].axvline(az, color="cyan", ls="--")
    axes[0].set_xlabel("Азимут, °")
    axes[0].set_ylabel("dB")
    axes[0].set_title("Срез по азимуту")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(d_m, spec_db[i_a, :])
    axes[1].axvline(dist, color="cyan", ls="--")
    if ground_truth:
        axes[1].axvline(ground_truth[1], color="lime", ls=":")
    axes[1].set_xlabel("d, м")
    axes[1].set_title("Срез по расстоянию")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"{stage}: одномерные срезы (планарный режим)")
    _savefig(os.path.join(out_dir, f"{prefix}{idx}_planar_1d_slices.png"))


def _plot_polar_azimuth(result: MusicResult, path: str) -> None:
    if result.spectrum_fine is None or result.grid_fine is None:
        return
    grid = result.grid_fine
    if grid.get("planar"):
        theta = np.deg2rad(grid["theta"])
        spec = result.spectrum_fine
        if spec.ndim == 1:
            slice_theta = 10 * np.log10(spec / (spec.max() + 1e-12) + 1e-12)
        else:
            d = grid["d"]
            i_d = int(np.argmin(np.abs(d - result.distance_m)))
            slice_theta = spec[:, i_d]
            slice_theta = 10 * np.log10(slice_theta / (slice_theta.max() + 1e-12) + 1e-12)
    else:
        theta = np.deg2rad(grid["theta"])
        phi = grid["phi"]
        d = grid["d"]
        spec = result.spectrum_fine
        i_phi = int(np.argmin(np.abs(phi - result.elevation_deg)))
        i_d = int(np.argmin(np.abs(d - result.distance_m)))
        slice_theta = spec[:, i_phi, i_d]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(theta, slice_theta / (slice_theta.max() + 1e-12), color="tab:orange", lw=2)
    ax.axvline(np.deg2rad(result.azimuth_deg), color="cyan", ls="--", label="пик θ")
    ax.set_title("Нормированный MUSIC-срез по азимуту (fine, фикс. φ и d)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    _savefig(path)


def _plot_summary_card(
    result: MusicResult,
    path: str,
    ground_truth: Optional[tuple[float, float]],
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    lines = [
        title or "MUSIC Localization",
        "",
        f"Азимут θ:     {result.azimuth_deg:.2f}°",
        f"Угол места φ: {result.elevation_deg:.2f}°",
        f"Расстояние d: {result.distance_m:.3f} м",
    ]
    if ground_truth:
        err_t = (result.azimuth_deg - ground_truth[0] + 180) % 360 - 180
        err_d = result.distance_m - ground_truth[1]
        lines.extend(
            [
                "",
                f"Эталон (файл): θ={ground_truth[0]:.0f}°, d={ground_truth[1]:.2f} м",
                f"Ошибка:        Δθ={err_t:.2f}°, Δd={err_d:.3f} м",
            ]
        )
    ax.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=13, family="monospace")
    _savefig(path)
