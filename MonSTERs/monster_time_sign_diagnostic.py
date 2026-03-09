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
OUTPUT_PATH = ROOT / "outputs" / "monster_time_sign_diagnostic.png"

IMAGE_SIZE = 16
EMBED_DIM = 768
THETA_BASE = 10_000.0
TOP_DELTA = 16.0
UNIT_SCALE = 1.0

QUERY_X = 8
QUERY_Y = 8
QUERY_ON_VALUE = 100.0

KEY_T_VALUE = 0.0
QUERY_T_ABS_VALUES = [8.0, 16.0, 24.0]

SEED = 0
METRIC = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float64)
METRIC_LABEL = "(-,+,+,+)"

FIGSIZE = (24.0, 4.8)
AXES_PAD = 0.08
CBAR_RATIO = 0.05


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


def metric_dot_batch(query_vec: np.ndarray, keys: np.ndarray, metric: np.ndarray) -> np.ndarray:
    query_metric = query_vec.reshape(-1, 4) @ metric
    return np.sum(keys.reshape(keys.shape[0], -1, 4) * query_metric[None, :, :], axis=(1, 2))


def transform_positions(
    base_vector: np.ndarray,
    positions: np.ndarray,
    monster: TriadMonSTERFastVec,
) -> np.ndarray:
    transformed = np.empty((positions.shape[0], monster.dim), dtype=np.float64)
    for idx, position in enumerate(positions):
        transformed[idx] = apply_monster_triad_fast_vec(
            base_vector,
            monster.forward(position),
            dim=monster.dim,
        )
    return transformed


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


def score_map(
    transformed_query: np.ndarray,
    transformed_keys: np.ndarray,
    metric: np.ndarray,
) -> np.ndarray:
    logits = metric_dot_batch(transformed_query, transformed_keys, metric) / np.sqrt(transformed_query.size)
    side = int(np.sqrt(transformed_keys.shape[0]))
    return logits.reshape(side, side)


def plot_grid(
    query_image: np.ndarray,
    positive_titles: list[str],
    negative_titles: list[str],
    positive_maps: list[np.ndarray],
    negative_maps: list[np.ndarray],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=FIGSIZE)
    score_norm = Normalize(
        vmin=min(float(panel.min()) for panel in positive_maps + negative_maps),
        vmax=max(float(panel.max()) for panel in positive_maps + negative_maps),
    )
    grid = GridSpec(
        1,
        8,
        figure=fig,
        left=0.03,
        right=0.97,
        top=0.74,
        bottom=0.16,
        wspace=AXES_PAD,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, CBAR_RATIO],
    )

    binary_ax = fig.add_subplot(grid[0, 0])
    positive_axes = [fig.add_subplot(grid[0, idx]) for idx in (1, 2, 3)]
    negative_axes = [fig.add_subplot(grid[0, idx]) for idx in (4, 5, 6)]
    cbar_ax = fig.add_subplot(grid[0, 7])

    fig.suptitle(
        (
            f"MonSTERs time-sign diagnostic | distinct q/k vectors | metric {METRIC_LABEL} | "
            f"dim={EMBED_DIM} | base={THETA_BASE:g} | top_delta={TOP_DELTA:g}"
        ),
        fontsize=15,
    )

    binary_ax.imshow(query_image, cmap="viridis", vmin=0.0, vmax=QUERY_ON_VALUE, interpolation="nearest")
    binary_ax.set_title("Binary input", fontsize=11)
    binary_ax.axis("off")

    last_image = None
    for axis, title, panel in zip(positive_axes, positive_titles, positive_maps):
        last_image = axis.imshow(panel, cmap="viridis", norm=score_norm)
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    for axis, title, panel in zip(negative_axes, negative_titles, negative_maps):
        last_image = axis.imshow(panel, cmap="viridis", norm=score_norm)
        axis.set_title(title, fontsize=11)
        axis.axis("off")

    if last_image is None:
        raise RuntimeError("Expected at least one score panel.")
    fig.colorbar(last_image, cax=cbar_ax)

    positive_left = positive_axes[0].get_position().x0
    positive_right = positive_axes[-1].get_position().x1
    negative_left = negative_axes[0].get_position().x0
    negative_right = negative_axes[-1].get_position().x1
    fig.text(
        0.5 * (positive_left + positive_right),
        0.80,
        "Query ahead of keys (+Δt)",
        ha="center",
        va="center",
        fontsize=13,
    )
    fig.text(
        0.5 * (negative_left + negative_right),
        0.80,
        "Query behind keys (-Δt)",
        ha="center",
        va="center",
        fontsize=13,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if EMBED_DIM % 12 != 0:
        raise ValueError("EMBED_DIM must be divisible by 12 for MonSTERs.")
    if not (0 <= QUERY_X < IMAGE_SIZE and 0 <= QUERY_Y < IMAGE_SIZE):
        raise ValueError("Query position must lie within the image grid.")

    rng = np.random.default_rng(SEED)
    query_base_vector = random_embedding(EMBED_DIM, rng)
    key_base_vector = random_embedding(EMBED_DIM, rng)

    query_index = QUERY_Y * IMAGE_SIZE + QUERY_X
    monster = TriadMonSTERFastVec(dim=EMBED_DIM, base=THETA_BASE, top_delta=TOP_DELTA)
    monster.unit = UNIT_SCALE / TOP_DELTA

    key_positions = make_standard_positions(IMAGE_SIZE, KEY_T_VALUE)
    transformed_keys = transform_positions(key_base_vector, key_positions, monster)
    query_position_base = key_positions[query_index].copy()

    positive_titles: list[str] = []
    negative_titles: list[str] = []
    positive_maps: list[np.ndarray] = []
    negative_maps: list[np.ndarray] = []

    for abs_t in QUERY_T_ABS_VALUES:
        positive_query_position = query_position_base.copy()
        positive_query_position[0] = abs_t
        positive_query = transform_vector_at_position(query_base_vector, positive_query_position, monster)
        positive_maps.append(score_map(positive_query, transformed_keys, METRIC))
        positive_titles.append(f"key t={KEY_T_VALUE:g}, q t=+{abs_t:g}")

        negative_query_position = query_position_base.copy()
        negative_query_position[0] = -abs_t
        negative_query = transform_vector_at_position(query_base_vector, negative_query_position, monster)
        negative_maps.append(score_map(negative_query, transformed_keys, METRIC))
        negative_titles.append(f"key t={KEY_T_VALUE:g}, q t=-{abs_t:g}")

    query_image = make_query_image(IMAGE_SIZE, QUERY_X, QUERY_Y, QUERY_ON_VALUE)
    plot_grid(query_image, positive_titles, negative_titles, positive_maps, negative_maps, OUTPUT_PATH)

    print(f"Saved time-sign diagnostic to {OUTPUT_PATH}")
    print(f"Metric: {METRIC_LABEL}")
    print(f"Key t value: {KEY_T_VALUE}")
    print(f"Query |t| values: {QUERY_T_ABS_VALUES}")
    print(f"Unit scale: {UNIT_SCALE:.6g} / top_delta")
    print("Distinct random vectors are used for query and key content.")
    for abs_t, positive_map, negative_map in zip(QUERY_T_ABS_VALUES, positive_maps, negative_maps):
        diff = positive_map - negative_map
        print(
            "abs_t={:.0f} | max_abs_diff={:.6g} | mean_abs_diff={:.6g}".format(
                abs_t,
                float(np.max(np.abs(diff))),
                float(np.mean(np.abs(diff))),
            )
        )


if __name__ == "__main__":
    main()
