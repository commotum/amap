from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

from tilted_plane import tilted_plane_coords
from v12 import ETA4, TriadMonSTERFastVec, apply_monster_triad_fast_vec


# ============================================================================
# Editable hyperparameters
# ============================================================================
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "outputs" / "monster_hyperparam_grid.png"

IMAGE_SIZE = 64
EMBED_DIM = 768
THETA_BASE = 10_000.0
TOP_DELTA = 64.0
SEED = 0

QUERY_X = 32
QUERY_Y = 32
QUERY_ON_VALUE = 100.0

T_VALUES = [0.0, 16.0, 32.0]

FIGSIZE = (24.0, 4.5)
AXES_PAD = 0.18
CBAR_SIZE = "5%"
CBAR_PAD = 0.18

ETA4_NEGPOS = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float64)


def random_embedding(d: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(0.0, 1.0, d)
    v = v / np.linalg.norm(v) * np.sqrt(d)
    return v


def make_query_image(size: int, query_x: int, query_y: int, on_value: float) -> np.ndarray:
    image = np.zeros((size, size), dtype=float)
    image[query_y, query_x] = on_value
    return image


def centered_xy_coords(size: int) -> tuple[np.ndarray, np.ndarray]:
    vals = np.arange(-(size / 2) + 0.5, size / 2, 1.0)
    return np.meshgrid(vals, vals, indexing="xy")


def make_standard_positions(size: int, t_value: float) -> np.ndarray:
    x, y = centered_xy_coords(size)
    t = np.full_like(x, fill_value=t_value, dtype=np.float64)
    z = np.zeros_like(x, dtype=np.float64)
    return np.stack((t, x, y, z), axis=-1).reshape(-1, 4)


def make_tilted_positions(size: int, t_value: float) -> np.ndarray:
    x, y, z = tilted_plane_coords(size=size)
    t = np.full_like(x, fill_value=t_value, dtype=np.float64)
    return np.stack((t, x, y, z), axis=-1).reshape(-1, 4)


def metric_dot_batch(query_vec: np.ndarray, keys: np.ndarray, metric: np.ndarray) -> np.ndarray:
    query_metric = query_vec.reshape(-1, 4) @ metric
    return np.sum(keys.reshape(keys.shape[0], -1, 4) * query_metric[None, :, :], axis=(1, 2))


def transform_positions(
    base_vector: np.ndarray,
    positions: np.ndarray,
    monster: TriadMonSTERFastVec,
) -> np.ndarray:
    transformed_keys = np.empty((positions.shape[0], monster.dim), dtype=np.float64)
    for idx, position in enumerate(positions):
        transformed_keys[idx] = apply_monster_triad_fast_vec(
            base_vector,
            monster.forward(position),
            dim=monster.dim,
        )
    return transformed_keys


def score_transformed_vectors(
    transformed_keys: np.ndarray,
    query_index: int,
    metric: np.ndarray,
) -> np.ndarray:
    transformed_query = transformed_keys[query_index]
    logits = metric_dot_batch(transformed_query, transformed_keys, metric) / np.sqrt(transformed_query.size)
    side = int(np.sqrt(transformed_keys.shape[0]))
    return logits.reshape(side, side)


def plot_grid(
    query_image: np.ndarray,
    panel_titles: list[str],
    top_row_maps: list[np.ndarray],
    bottom_row_maps: list[np.ndarray],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(24.0, 8.4))
    top_grid = ImageGrid(
        fig,
        (0.03, 0.53, 0.94, 0.30),
        nrows_ncols=(1, 7),
        axes_pad=AXES_PAD,
    )
    bottom_grid = ImageGrid(
        fig,
        (0.03, 0.10, 0.94, 0.30),
        nrows_ncols=(1, 7),
        axes_pad=AXES_PAD,
    )
    top_axes = list(top_grid)
    bottom_axes = list(bottom_grid)

    fig.suptitle(
        (
            f"MonSTERs comparison | dim={EMBED_DIM} | base={THETA_BASE:g} | "
            f"top_delta={TOP_DELTA:g} | query=({QUERY_X}, {QUERY_Y})"
        ),
        fontsize=15,
    )

    for axis in (top_axes[0], bottom_axes[0]):
        axis.imshow(query_image, cmap="viridis", vmin=0.0, vmax=QUERY_ON_VALUE, interpolation="nearest")
        axis.axis("off")
    top_axes[0].set_title("Binary input", fontsize=11)
    bottom_axes[0].set_title("Binary input", fontsize=11)

    for axis, title, panel in zip(top_axes[1:], panel_titles, top_row_maps):
        axis.imshow(panel, cmap="viridis", vmin=float(panel.min()), vmax=float(panel.max()))
        axis.set_title(title, fontsize=11)
        axis.axis("off")

    for axis, panel in zip(bottom_axes[1:], bottom_row_maps):
        axis.imshow(panel, cmap="viridis", vmin=float(panel.min()), vmax=float(panel.max()))
        axis.axis("off")

    fig.text(0.018, 0.68, "(+,-,-,-)", rotation=90, va="center", ha="center", fontsize=13)
    fig.text(0.018, 0.26, "(-,+,+,+)", rotation=90, va="center", ha="center", fontsize=13)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if EMBED_DIM % 12 != 0:
        raise ValueError("EMBED_DIM must be divisible by 12 for MonSTERs.")
    if not (0 <= QUERY_X < IMAGE_SIZE and 0 <= QUERY_Y < IMAGE_SIZE):
        raise ValueError("Query position must lie within the image grid.")

    rng = np.random.default_rng(SEED)
    base_vector = random_embedding(EMBED_DIM, rng)
    query_index = QUERY_Y * IMAGE_SIZE + QUERY_X
    monster = TriadMonSTERFastVec(dim=EMBED_DIM, base=THETA_BASE, top_delta=TOP_DELTA)

    panel_titles: list[str] = []
    transformed_panels: list[np.ndarray] = []

    for t_value in T_VALUES:
        positions = make_standard_positions(IMAGE_SIZE, t_value)
        transformed_panels.append(transform_positions(base_vector, positions, monster))
        panel_titles.append(f"Std t={t_value:g}, z=0")

    for t_value in T_VALUES:
        positions = make_tilted_positions(IMAGE_SIZE, t_value)
        transformed_panels.append(transform_positions(base_vector, positions, monster))
        panel_titles.append(f"Tilted t={t_value:g}")

    top_row_maps = [score_transformed_vectors(panel, query_index, ETA4) for panel in transformed_panels]
    bottom_row_maps = [score_transformed_vectors(panel, query_index, ETA4_NEGPOS) for panel in transformed_panels]

    query_image = make_query_image(IMAGE_SIZE, QUERY_X, QUERY_Y, QUERY_ON_VALUE)
    plot_grid(query_image, panel_titles, top_row_maps, bottom_row_maps, OUTPUT_PATH)

    print(f"Saved MonSTERs comparison grid to {OUTPUT_PATH}")
    print(f"T values: {T_VALUES}")
    print("First three panels use standard centered (x, y, z=0) coordinates.")
    print("Last three panels use tilted-plane coordinates from MonSTERs/tilted_plane.py.")
    print("Top row uses metric (+,-,-,-); bottom row uses metric (-,+,+,+).")


if __name__ == "__main__":
    main()
