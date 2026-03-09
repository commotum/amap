from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


@dataclass(frozen=True)
class AttentionConfig:
    image_size: int = 64
    embed_dim: int = 512
    num_directions: int = 8
    theta_base: float = 10_000.0
    seed: int = 0
    query_x: int = 32
    query_y: int = 32


def random_embedding(d: int = 512, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    v = rng.normal(0.0, 1.0, d)
    v = v / np.linalg.norm(v) * np.sqrt(d)
    return v


def rotate_pairs(pairs: np.ndarray, angles: np.ndarray) -> np.ndarray:
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    x0 = pairs[:, 0]
    x1 = pairs[:, 1]
    rotated = np.empty_like(pairs)
    rotated[:, 0] = cos_angles * x0 - sin_angles * x1
    rotated[:, 1] = sin_angles * x0 + cos_angles * x1
    return rotated


def base_frequencies(num_freqs: int, theta_base: float) -> np.ndarray:
    return theta_base ** (-np.arange(num_freqs) / num_freqs)


def axial_encode(vector: np.ndarray, x: int, y: int, theta_base: float) -> np.ndarray:
    d = vector.size
    if d % 4 != 0:
        raise ValueError("Axial RoPE requires embed_dim divisible by 4.")

    num_freqs = d // 4
    freqs = base_frequencies(num_freqs, theta_base)

    x_half = vector[: d // 2].reshape(num_freqs, 2)
    y_half = vector[d // 2 :].reshape(num_freqs, 2)

    x_rot = rotate_pairs(x_half, x * freqs).reshape(-1)
    y_rot = rotate_pairs(y_half, y * freqs).reshape(-1)
    return np.concatenate((x_rot, y_rot))


def spiral_frequency_sets(embed_dim: int, num_directions: int, theta_base: float) -> list[np.ndarray]:
    if num_directions % 2 != 0:
        raise ValueError("Spiral RoPE requires an even number of directions.")
    if embed_dim % num_directions != 0:
        raise ValueError("Spiral RoPE requires embed_dim divisible by num_directions.")
    if (embed_dim // num_directions) % 2 != 0:
        raise ValueError("Each Spiral RoPE direction group must contain an even number of channels.")

    num_freqs = embed_dim // 4
    group_pairs = embed_dim // (2 * num_directions)
    freqs = base_frequencies(num_freqs, theta_base)
    assignments: list[np.ndarray] = []

    for direction_idx in range(num_directions):
        pair_idx = direction_idx % (num_directions // 2)
        freq_indices: list[int] = []
        for start in range(2 * pair_idx, num_freqs, num_directions):
            freq_indices.append(start)
            if start + 1 < num_freqs:
                freq_indices.append(start + 1)

        freq_set = freqs[np.asarray(freq_indices[:group_pairs], dtype=int)]
        if freq_set.size != group_pairs:
            raise ValueError("Frequency assignment does not match the required Spiral RoPE group size.")
        assignments.append(freq_set)

    return assignments


def spiral_encode(
    vector: np.ndarray,
    x: int,
    y: int,
    num_directions: int,
    theta_base: float,
    frequency_sets: list[np.ndarray],
) -> np.ndarray:
    d = vector.size
    group_dim = d // num_directions
    groups = vector.reshape(num_directions, group_dim)
    encoded_groups: list[np.ndarray] = []

    for direction_idx in range(num_directions):
        angle = direction_idx * np.pi / num_directions
        projected_position = x * np.cos(angle) + y * np.sin(angle)
        pairs = groups[direction_idx].reshape(-1, 2)
        rotated = rotate_pairs(pairs, projected_position * frequency_sets[direction_idx]).reshape(-1)
        encoded_groups.append(rotated)

    return np.concatenate(encoded_groups)


def compute_attention_map(
    base_vector: np.ndarray,
    positions: list[tuple[int, int]],
    query_index: int,
    encoder: str,
    config: AttentionConfig,
) -> np.ndarray:
    keys = np.empty((len(positions), config.embed_dim), dtype=float)

    if encoder == "axial":
        query = axial_encode(base_vector, *positions[query_index], theta_base=config.theta_base)
        for idx, (x, y) in enumerate(positions):
            keys[idx] = axial_encode(base_vector, x, y, theta_base=config.theta_base)
    elif encoder == "spiral":
        frequency_sets = spiral_frequency_sets(config.embed_dim, config.num_directions, config.theta_base)
        query = spiral_encode(
            base_vector,
            *positions[query_index],
            num_directions=config.num_directions,
            theta_base=config.theta_base,
            frequency_sets=frequency_sets,
        )
        for idx, (x, y) in enumerate(positions):
            keys[idx] = spiral_encode(
                base_vector,
                x,
                y,
                num_directions=config.num_directions,
                theta_base=config.theta_base,
                frequency_sets=frequency_sets,
            )
    else:
        raise ValueError(f"Unknown encoder: {encoder}")

    logits = keys @ query / np.sqrt(config.embed_dim)
    return logits.reshape(config.image_size, config.image_size)


def embedding_to_color(vector: np.ndarray) -> np.ndarray:
    rgb = vector[:3]
    rgb = 0.5 + 0.5 * np.tanh(rgb / 2.0)
    return np.clip(rgb, 0.0, 1.0)


def make_query_image(config: AttentionConfig, on_value: float = 100.0) -> np.ndarray:
    image = np.zeros((config.image_size, config.image_size), dtype=float)
    image[config.query_y, config.query_x] = on_value
    return image


def plot_results(
    image: np.ndarray,
    query_x: int,
    query_y: int,
    axial_logits: np.ndarray,
    spiral_logits: np.ndarray,
    output_path: Path,
) -> None:
    heat_min = min(float(axial_logits.min()), float(spiral_logits.min()))
    heat_max = max(float(axial_logits.max()), float(spiral_logits.max()))
    heat_norm = Normalize(vmin=heat_min, vmax=heat_max)

    fig = plt.figure(figsize=(13, 4))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 0.005, 1, 0.005, 1, 0.005])
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 4]),
        fig.add_subplot(gs[0, 5]),
    ]
    fig.suptitle(f"Query pixel: ({query_x}, {query_y})", fontsize=14)

    axes[0].imshow(image, cmap="viridis", vmin=0.0, vmax=100.0, interpolation="nearest")
    axes[0].set_title("Binary input")
    axes[0].axis("off")

    im01 = axes[1].imshow(axial_logits, cmap="viridis", norm=heat_norm)
    axes[1].set_title("Axial RoPE")
    axes[1].axis("off")

    im02 = axes[2].imshow(spiral_logits, cmap="viridis", norm=heat_norm)
    axes[2].set_title("Spiral RoPE")
    axes[2].axis("off")
    fig.colorbar(im02, cax=axes[3])

    fig.subplots_adjust(left=0.04, right=0.96, top=0.82, bottom=0.08, wspace=0.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize positional attention induced by Axial RoPE vs Spiral RoPE on a constant image."
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--embed-dims", type=int, nargs="+", default=[1024, 8096])
    parser.add_argument("--num-directions", type=int, default=8)
    parser.add_argument("--theta-base", type=float, default=10_000.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query-x", type=int, default=32)
    parser.add_argument("--query-y", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
    )
    args = parser.parse_args()
    if not (0 <= args.query_x < args.image_size and 0 <= args.query_y < args.image_size):
        parser.error("query position must lie within the image grid.")
    for embed_dim in args.embed_dims:
        if embed_dim % 4 != 0:
            parser.error("each embed dimension must be divisible by 4.")
        if embed_dim % args.num_directions != 0:
            parser.error("each embed dimension must be divisible by --num-directions.")
    return args


def run_for_embed_dim(args: argparse.Namespace, embed_dim: int) -> Path:
    config = AttentionConfig(
        image_size=args.image_size,
        embed_dim=embed_dim,
        num_directions=args.num_directions,
        theta_base=args.theta_base,
        seed=args.seed,
        query_x=args.query_x,
        query_y=args.query_y,
    )

    rng = np.random.default_rng(config.seed)
    input_embedding = random_embedding(config.embed_dim, rng)
    positions = [(x, y) for y in range(config.image_size) for x in range(config.image_size)]
    query_index = config.query_y * config.image_size + config.query_x

    axial_logits = compute_attention_map(
        input_embedding,
        positions,
        query_index,
        encoder="axial",
        config=config,
    )
    spiral_logits = compute_attention_map(
        input_embedding,
        positions,
        query_index,
        encoder="spiral",
        config=config,
    )

    image = make_query_image(config)
    plot_results(
        image,
        config.query_x,
        config.query_y,
        axial_logits,
        spiral_logits,
        args.output_dir / f"rope_attention_single_color_{embed_dim}d.png",
    )

    output_path = args.output_dir / f"rope_attention_single_color_{embed_dim}d.png"
    print(f"Saved attention visualization to {output_path}")
    print(f"Embedding dim: {embed_dim}")
    print(f"Query pixel: ({config.query_x}, {config.query_y})")
    print("Attention uses the same constant embedding for queries and keys at every pixel.")
    return output_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for embed_dim in args.embed_dims:
        run_for_embed_dim(args, embed_dim)
        print("---")


if __name__ == "__main__":
    main()
