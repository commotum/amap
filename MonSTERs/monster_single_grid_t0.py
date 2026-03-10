from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import numpy as np

from v12 import TriadMonSTERFastVec, apply_monster_triad_fast_vec


# ============================================================================
# Editable hyperparameters
# ============================================================================
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "outputs" / "monster_single_grid_t0.png"

IMAGE_SIZE = 17
EMBED_DIM = 768
THETA_BASE = 10_000.0
TOP_DELTA = 16.0
SEED = 0

QUERY_X = 8
QUERY_Y = 8
KEY_T_VALUE = 0.0
QUERY_T_VALUE = 0.0
UNIT_SCALE = 1.0

FIGSIZE = (4.4, 4.8)
CBAR_RATIO = 0.06

ETA4_NEGPOS = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float64)


def random_embedding(d: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(0.0, 1.0, d)
    v = v / np.linalg.norm(v) * np.sqrt(d)
    return v


def centered_xy_coords(size: int) -> tuple[np.ndarray, np.ndarray]:
    vals = np.arange(-(size / 2) + 0.5, size / 2, 1.0)
    return np.meshgrid(vals, vals, indexing="xy")


def make_standard_positions(size: int, t_value: float) -> np.ndarray:
    x, y = centered_xy_coords(size)
    t = np.full_like(x, fill_value=t_value, dtype=np.float64)
    z = np.zeros_like(x, dtype=np.float64)
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


def transform_vector_at_position(
    base_vector: np.ndarray,
    position: np.ndarray,
    monster: TriadMonSTERFastVec,
) -> np.ndarray:
    return apply_monster_triad_fast_vec(
        base_vector,
        monster.forward(position),
        dim=monster.dim,
    )


def score_transformed_vectors(
    transformed_query: np.ndarray,
    transformed_keys: np.ndarray,
    metric: np.ndarray,
) -> np.ndarray:
    logits = metric_dot_batch(transformed_query, transformed_keys, metric) / np.sqrt(transformed_query.size)
    side = int(np.sqrt(transformed_keys.shape[0]))
    return logits.reshape(side, side)


def plot_grid(score_map: np.ndarray, output_path: Path) -> None:
    fig = plt.figure(figsize=FIGSIZE)
    grid = GridSpec(
        1,
        2,
        figure=fig,
        left=0.06,
        right=0.94,
        top=0.96,
        bottom=0.06,
        wspace=0.08,
        width_ratios=[1.0, CBAR_RATIO],
    )
    axis = fig.add_subplot(grid[0, 0])
    cbar_axis = fig.add_subplot(grid[0, 1])

    image = axis.imshow(
        score_map,
        cmap="viridis",
        norm=Normalize(vmin=float(score_map.min()), vmax=float(score_map.max())),
    )
    axis.axis("off")
    fig.colorbar(image, cax=cbar_axis)

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
    monster.unit = UNIT_SCALE / TOP_DELTA

    key_positions = make_standard_positions(IMAGE_SIZE, KEY_T_VALUE)
    transformed_keys = transform_positions(base_vector, key_positions, monster)

    query_position = key_positions[query_index].copy()
    query_position[0] = QUERY_T_VALUE
    transformed_query = transform_vector_at_position(base_vector, query_position, monster)

    score_map = score_transformed_vectors(transformed_query, transformed_keys, ETA4_NEGPOS)
    plot_grid(score_map, OUTPUT_PATH)

    print(f"Saved single-grid MonSTER plot to {OUTPUT_PATH}")
    print(f"Key t value: {KEY_T_VALUE}")
    print(f"Query t value: {QUERY_T_VALUE}")
    print("The plot uses metric (-,+,+,+) with no titles or labels.")


if __name__ == "__main__":
    main()
