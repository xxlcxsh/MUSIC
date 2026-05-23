"""
sound_direction.py — определение угла прихода звука (GCC-PHAT / SRP-PHAT)
==========================================================================
8 микрофонов на окружности диаметром 1 м.
Читает Audacity .aup / .aup3, вычисляет угол методом SRP-PHAT,
рисует схему с микрофонами и стрелкой направления.

Запуск:
    python sound_direction.py "channels8/A1_S1_25_10.aup" --plot
    python sound_direction.py "channels8/A1_S1_25_10.aup" --block-len 44100
"""

import argparse, os, sys, struct, warnings
import xml.etree.ElementTree as ET
import numpy as np

try:
    import soundfile as sf;      HAS_SF  = True
except ImportError:              HAS_SF  = False
try:
    import sqlite3;              HAS_DB  = True
except ImportError:              HAS_DB  = False
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:              HAS_MPL = False

# ── геометрия массива ─────────────────────────────────────────────────────────
N_MICS     = 8
RADIUS     = 0.5           # метров (диаметр 1 м)
MIC_ANGLES = np.linspace(0, 2 * np.pi, N_MICS, endpoint=False)
MIC_POS    = np.column_stack([RADIUS * np.cos(MIC_ANGLES),
                               RADIUS * np.sin(MIC_ANGLES)])

# ─────────────────────────────────────────────────────────────────────────────
#  ШАГ 1. GCC-PHAT — возвращает ВЕСЬ вектор корреляции и лаги
# ─────────────────────────────────────────────────────────────────────────────
def gcc_phat_full(sig1: np.ndarray, sig2: np.ndarray, fs: float, max_delay: float):
    """
    Возвращает (lags, gcc) — вектор лагов (сек) и значения GCC-PHAT.
    Ограничение поиска: |lag| <= max_delay.
    """
    n    = len(sig1) + len(sig2) - 1
    nfft = 1 << (n - 1).bit_length()

    S1 = np.fft.rfft(sig1, n=nfft)
    S2 = np.fft.rfft(sig2, n=nfft)
    R  = S1 * np.conj(S2)

    denom = np.abs(R)
    denom[denom < 1e-10] = 1e-10
    R /= denom                              # PHAT-взвешивание

    gcc  = np.fft.fftshift(np.fft.irfft(R, n=nfft))
    lags = np.arange(-(nfft // 2), nfft - nfft // 2) / fs

    mask = np.abs(lags) <= max_delay
    return lags[mask], gcc[mask]


# ─────────────────────────────────────────────────────────────────────────────
#  ШАГ 2. SRP-PHAT — суммируем GCC по всем парам для каждого угла-кандидата
# ─────────────────────────────────────────────────────────────────────────────
def srp_phat_angle(segments: list, fs: float, c: float = 343.0,
                   angle_resolution: float = 0.5) -> tuple:
    """
    Steered Response Power — PHAT для кругового массива.

    Для каждого угла θ (дальнее поле) ожидаемый TDOA между mic_i и mic_j:
        τ_ij(θ) = -( (xi-xj)·cos(θ) + (yi-yj)·sin(θ) ) / c

    Суммируем GCC_ij( τ_ij(θ) ) по всем парам → пик = направление звука.

    Возвращает (angles_deg, power, best_angle_deg).
    """
    max_delay = 2 * RADIUS / c          # физически максимальный TDOA

    # Заранее вычислим GCC для всех пар (i, j), i < j
    pairs   = [(i, j) for i in range(N_MICS) for j in range(i+1, N_MICS)]
    gcc_all = {}
    for (i, j) in pairs:
        lags, gcc = gcc_phat_full(segments[i], segments[j], fs, max_delay)
        gcc_all[(i, j)] = (lags, gcc)

    # Сетка углов-кандидатов
    angles_rad = np.deg2rad(np.arange(0, 360, angle_resolution))
    power      = np.zeros(len(angles_rad))

    for k, theta in enumerate(angles_rad):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        acc = 0.0
        for (i, j), (lags, gcc) in gcc_all.items():
            xi, yi = MIC_POS[i]
            xj, yj = MIC_POS[j]
            # ожидаемый TDOA (дальнее поле)
            tau_expected = -((xi - xj) * cos_t + (yi - yj) * sin_t) / c
            # интерполируем GCC в точке tau_expected
            acc += float(np.interp(tau_expected, lags, gcc))
        power[k] = acc

    best_idx       = int(np.argmax(power))
    best_angle_deg = float(np.degrees(angles_rad[best_idx]))
    angles_deg     = np.degrees(angles_rad)

    return angles_deg, power, best_angle_deg


# ─────────────────────────────────────────────────────────────────────────────
#  Чтение .au блоков Audacity
# ─────────────────────────────────────────────────────────────────────────────
def _read_au_block(path: str) -> np.ndarray:
    # Стратегия 1: soundfile (самый надёжный)
    if HAS_SF:
        try:
            data, _ = sf.read(path, always_2d=False, dtype='float32')
            return data.astype(np.float32)
        except Exception:
            pass

    with open(path, 'rb') as f:
        raw = f.read()
    if len(raw) < 4:
        return np.zeros(0, dtype=np.float32)

    SND = 0x2e736e64
    dtype_map = {2: np.int8, 3: np.int16, 5: np.int32, 6: np.float32, 7: np.float64}

    def _parse(endian):
        mg, offset, _, enc, _, _ = struct.unpack_from(f'{endian}IIIIII', raw)
        if mg != SND or enc not in dtype_map:
            raise ValueError
        dt      = np.dtype(dtype_map[enc]).newbyteorder(endian)
        samples = np.frombuffer(raw, dtype=dt, offset=offset)
        if samples.dtype.kind == 'i':
            samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
        return samples.astype(np.float32)

    # Стратегия 2 & 3: big/little endian Sun AU
    for endian in ('>', '<'):
        try:
            return _parse(endian)
        except Exception:
            pass

    # Стратегия 4: raw int16
    warnings.warn(f"Fallback raw int16: {os.path.basename(path)}")
    return (np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)


def _load_track(block_files: list) -> np.ndarray:
    return np.concatenate([_read_au_block(p) for p in block_files])


# ─────────────────────────────────────────────────────────────────────────────
#  Загрузка AUP (XML)
# ─────────────────────────────────────────────────────────────────────────────
def load_aup_xml(aup_path: str):
    tree = ET.parse(aup_path)
    root = tree.getroot()
    rate     = float(root.get('rate', 44100))
    base_dir = os.path.dirname(os.path.abspath(aup_path))
    uri      = root.tag.split('}')[0].lstrip('{') if '}' in root.tag else ''

    def tag(name):
        return f'{{{uri}}}{name}' if uri else name

    tracks = []
    for wavetrack in root.iter(tag('wavetrack')):
        blocks = {}
        for seq in wavetrack.iter(tag('sequence')):
            for blk in seq.iter(tag('simpleblockfile')):
                fname  = blk.get('filename')
                offset = int(blk.get('start', 0))
                if fname:
                    for dirpath, _, files in os.walk(base_dir):
                        if fname in files:
                            blocks[offset] = os.path.join(dirpath, fname)
                            break
        if blocks:
            tracks.append(_load_track([blocks[k] for k in sorted(blocks)]))

    if not tracks:
        raise RuntimeError(
            "Треки не найдены. Убедитесь что папка _data находится рядом с .aup файлом.")
    return tracks, rate


# ─────────────────────────────────────────────────────────────────────────────
#  Загрузка AUP3 (SQLite)
# ─────────────────────────────────────────────────────────────────────────────
def load_aup3_sqlite(path: str):
    if not HAS_DB:
        raise ImportError("sqlite3 недоступен")
    con = sqlite3.connect(path)
    cur = con.cursor()
    try:
        cur.execute("SELECT dict FROM project LIMIT 1")
        import json
        rate = float(json.loads(cur.fetchone()[0]).get('rate', 44100))
    except Exception:
        rate = 44100.0
    cur.execute("SELECT DISTINCT trackid FROM sampleblocks ORDER BY trackid")
    tracks = []
    for (tid,) in cur.fetchall():
        cur.execute("SELECT samples FROM sampleblocks WHERE trackid=? ORDER BY blockid", (tid,))
        parts = [np.frombuffer(blob, dtype='<f4') for (blob,) in cur.fetchall()]
        if parts:
            tracks.append(np.concatenate(parts))
    con.close()
    if not tracks:
        raise RuntimeError("Нет блоков в .aup3 файле")
    return tracks, rate


def load_project(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.aup3':
        return load_aup3_sqlite(path)
    if ext == '.aup':
        return load_aup_xml(path)
    if HAS_SF:
        data, sr = sf.read(path, always_2d=True)
        return [data[:, ch] for ch in range(data.shape[1])], float(sr)
    raise ValueError(f"Неизвестный формат: {ext}")


# ─────────────────────────────────────────────────────────────────────────────
#  Визуализация
# ─────────────────────────────────────────────────────────────────────────────
def plot_direction(best_angle_deg: float, angles_deg: np.ndarray,
                   power: np.ndarray):
    if not HAS_MPL:
        print("matplotlib не установлен — пропускаем график.")
        return

    fig = plt.figure(figsize=(14, 6), facecolor='#0d1117')
    fig.suptitle('Sound Direction Finder  —  GCC-PHAT / SRP-PHAT',
                 color='#e6edf3', fontsize=14, y=0.97)

    # ── левый: схема массива ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_facecolor('#0d1117')
    ax1.set_aspect('equal')
    ax1.set_title('Расположение микрофонов и направление звука',
                  color='#8b949e', fontsize=10, pad=10)

    # сетка
    for r in [0.5, 1.0, 1.5, 2.0]:
        c = plt.Circle((0, 0), r, color='#21262d', fill=False, lw=0.6, ls='--')
        ax1.add_patch(c)
    ax1.axhline(0, color='#21262d', lw=0.5)
    ax1.axvline(0, color='#21262d', lw=0.5)

    # окружность массива
    ring = plt.Circle((0, 0), RADIUS, color='#30363d', fill=False, lw=1.5)
    ax1.add_patch(ring)

    # микрофоны
    for i, (mx, my) in enumerate(MIC_POS):
        ax1.scatter(mx, my, s=120, color='#58a6ff', zorder=6,
                    edgecolors='#0d1117', linewidths=1.5)
        offset = np.array([mx, my]) * 0.25
        ax1.annotate(f'M{i}', (mx + offset[0], my + offset[1]),
                     color='#8b949e', fontsize=8, ha='center', va='center')

    # центр массива
    ax1.scatter(0, 0, s=30, color='#58a6ff', zorder=5, marker='+')

    # стрелка направления звука
    theta = np.deg2rad(best_angle_deg)
    arrow_len = 1.6
    dx, dy = arrow_len * np.cos(theta), arrow_len * np.sin(theta)
    ax1.annotate('', xy=(dx, dy), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='#f78166',
                                 lw=2.5, mutation_scale=20))

    # полоса неопределённости ±5°
    for offset_deg in (-5, 5):
        t2 = np.deg2rad(best_angle_deg + offset_deg)
        ax1.plot([0, 1.6 * np.cos(t2)], [0, 1.6 * np.sin(t2)],
                 color='#f78166', alpha=0.2, lw=1)

    # подпись угла
    ax1.text(dx * 0.5, dy * 0.5 + 0.12,
             f'{best_angle_deg:.1f}°',
             color='#ffa657', fontsize=11, fontweight='bold',
             ha='center', va='bottom')

    ax1.set_xlim(-2.2, 2.2)
    ax1.set_ylim(-2.2, 2.2)
    ax1.set_xlabel('X (м)', color='#8b949e')
    ax1.set_ylabel('Y (м)', color='#8b949e')
    ax1.tick_params(colors='#8b949e')
    for sp in ax1.spines.values():
        sp.set_edgecolor('#30363d')

    # легенда
    ax1.legend(handles=[
        mpatches.Patch(color='#58a6ff', label='Микрофон'),
        mpatches.Patch(color='#f78166', label=f'Направление {best_angle_deg:.1f}°'),
    ], loc='lower right', facecolor='#21262d', edgecolor='#30363d',
       labelcolor='#e6edf3', fontsize=8)

    # ── правый: полярная диаграмма мощности SRP-PHAT ─────────────────────────
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')
    ax2.set_facecolor('#0d1117')
    ax2.set_title('SRP-PHAT — мощность по углу',
                  color='#8b949e', fontsize=10, pad=15)

    power_norm = (power - power.min()) / (power.max() - power.min() + 1e-12)
    theta_rad  = np.deg2rad(angles_deg)

    # заполнение под кривой
    ax2.fill(theta_rad, power_norm, alpha=0.15, color='#58a6ff')
    ax2.plot(theta_rad, power_norm, color='#58a6ff', lw=1.2)

    # маркер максимума
    best_rad = np.deg2rad(best_angle_deg)
    ax2.plot([best_rad, best_rad], [0, 1.0], color='#f78166', lw=2, zorder=5)
    ax2.scatter([best_rad], [1.0], s=80, color='#f78166', zorder=6)

    ax2.set_theta_zero_location('E')
    ax2.set_theta_direction(1)
    ax2.tick_params(colors='#8b949e', labelsize=8)
    ax2.set_rlabel_position(45)
    ax2.grid(color='#21262d', linewidth=0.5)
    for sp in ax2.spines.values():
        sp.set_edgecolor('#30363d')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = 'direction_result.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"  График сохранён → {out}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Основная функция
# ─────────────────────────────────────────────────────────────────────────────
def run(aup_path: str, c: float = 343.0, block_start: int = 0,
        block_len: int = 44100, show_plot: bool = True):

    print(f"\n{'─'*55}")
    print("  GCC-PHAT / SRP-PHAT  —  определение угла звука")
    print(f"{'─'*55}")
    print(f"  Файл     : {aup_path}")
    print(f"  Блок     : [{block_start}, {block_start + block_len}] сэмплов")
    print(f"  Скорость звука: {c} м/с")
    print(f"{'─'*55}\n")

    # загрузка
    print("Загружаю треки …")
    tracks, fs = load_project(aup_path)
    print(f"  Треков: {len(tracks)}  |  fs = {fs:.0f} Гц")

    if len(tracks) < N_MICS:
        raise RuntimeError(f"Нужно {N_MICS} треков, найдено {len(tracks)}")

    # обрезаем до нужного окна
    segs = []
    for t in tracks[:N_MICS]:
        end = min(block_start + block_len, len(t))
        seg = t[block_start:end].astype(np.float64)
        if len(seg) < block_len:
            seg = np.pad(seg, (0, block_len - len(seg)))
        segs.append(seg)

    # SRP-PHAT
    print(f"Вычисляю SRP-PHAT по {N_MICS*(N_MICS-1)//2} парам микрофонов …")
    angles_deg, power, best_angle = srp_phat_angle(segs, fs, c,
                                                    angle_resolution=0.5)

    print(f"\n{'═'*55}")
    print(f"  Направление звука : {best_angle:.1f}°")
    print(f"  (0° = правее центра массива, против часовой стрелки)")
    print(f"{'═'*55}\n")

    if show_plot:
        plot_direction(best_angle, angles_deg, power)

    return best_angle



# ─────────────────────────────────────────────────────────────────────────────
# ╔══════════════════════════════════════════════════════╗
# ║            НАСТРОЙКИ — меняйте здесь                ║
# ╚══════════════════════════════════════════════════════╝

AUP_FILE    = r"E:\repos\music\MUSIC\audio\A1_S1_10_40.aup"
SPEED_SOUND = 343.0   # скорость звука, м/с
BLOCK_START = 0       # с какого сэмпла начинать анализ
BLOCK_LEN   = 44100   # сколько сэмплов анализировать (44100 = 1 секунда)
SHOW_PLOT   = True    # True = показать график

# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Консоль с аргументами — используем их
    # Sublime (Ctrl+B) без аргументов — используем настройки выше
    if len(sys.argv) > 1:
        p = argparse.ArgumentParser()
        p.add_argument('aup_file')
        p.add_argument('--speed-of-sound', type=float, default=343.0)
        p.add_argument('--block-start',    type=int,   default=0)
        p.add_argument('--block-len',      type=int,   default=44100)
        p.add_argument('--no-plot',        action='store_true')
        args = p.parse_args()
        run(aup_path    = args.aup_file,
            c           = args.speed_of_sound,
            block_start = args.block_start,
            block_len   = args.block_len,
            show_plot   = not args.no_plot)
    else:
        # Sublime Text: Ctrl+B запускает отсюда
        run(aup_path    = AUP_FILE,
            c           = SPEED_SOUND,
            block_start = BLOCK_START,
            block_len   = BLOCK_LEN,
            show_plot   = SHOW_PLOT)

if __name__ == '__main__':
    main()
