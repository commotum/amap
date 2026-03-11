from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ReconstructionConfig:
    grid_size: int = 64
    rope_dim: int = 1024
    theta_base: float = 10_000.0
    num_directions: int = 8
    circle_radius: float = 15.0


def make_point_image(size: int) -> np.ndarray:
    image = np.zeros((size, size), dtype=float)
    image[size // 2, size // 2] = 1.0
    return image


def make_circle_image(size: int, radius: float) -> np.ndarray:
    yy, xx = np.indices((size, size))
    cx = size // 2
    cy = size // 2
    return (((xx - cx) ** 2 + (yy - cy) ** 2) <= radius**2).astype(float)


def rope_frequencies(config: ReconstructionConfig) -> np.ndarray:
    num_freqs = config.rope_dim // 4
    return config.theta_base ** (-np.arange(num_freqs) / num_freqs)


def fft_angular_grid(size: int) -> np.ndarray:
    return np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(size))


def nearest_fft_bin(wx: float, wy: float, wgrid: np.ndarray) -> tuple[int, int]:
    ix = int(np.argmin(np.abs(wgrid - wx)))
    iy = int(np.argmin(np.abs(wgrid - wy)))
    return iy, ix


def build_axial_mask(config: ReconstructionConfig, theta: np.ndarray, wgrid: np.ndarray) -> np.ndarray:
    size = config.grid_size
    mask = np.zeros((size, size), dtype=bool)
    mask[size // 2, size // 2] = True

    for th in theta:
        for wx, wy in ((th, 0.0), (-th, 0.0), (0.0, th), (0.0, -th)):
            iy, ix = nearest_fft_bin(wx, wy, wgrid)
            mask[iy, ix] = True

    return mask


def build_spiral_mask(config: ReconstructionConfig, theta: np.ndarray, wgrid: np.ndarray) -> np.ndarray:
    size = config.grid_size
    k = config.num_directions
    if k % 2 != 0:
        raise ValueError("num_directions must be even for perpendicular direction pairing.")

    mask = np.zeros((size, size), dtype=bool)
    mask[size // 2, size // 2] = True

    phi = [direction * np.pi / k for direction in range(k)]

    # Eq. (13): adjacent frequency pairs are assigned round-robin to perpendicular pairs.
    for j in range(len(theta) // 2):
        pair_idx = j % (k // 2)
        assigned_dirs = (pair_idx, pair_idx + k // 2)
        th_pair = (theta[2 * j], theta[2 * j + 1])

        for dir_idx in assigned_dirs:
            ang = phi[dir_idx]
            cos_ang = np.cos(ang)
            sin_ang = np.sin(ang)

            for th in th_pair:
                for sign in (1.0, -1.0):
                    wx = sign * th * cos_ang
                    wy = sign * th * sin_ang
                    iy, ix = nearest_fft_bin(wx, wy, wgrid)
                    mask[iy, ix] = True

    return mask


def reconstruct(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    masked_spectrum = spectrum * mask
    return np.fft.ifft2(np.fft.ifftshift(masked_spectrum)).real


def plot_reconstructions(
    point_image: np.ndarray,
    point_axial: np.ndarray,
    point_spiral: np.ndarray,
    circle_image: np.ndarray,
    circle_axial: np.ndarray,
    circle_spiral: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    images = [
        (point_image, "Input"),
        (point_axial, "FFT-iFFT (Axial RoPE)"),
        (point_spiral, "FFT-iFFT (Spiral RoPE)"),
        (circle_image, "Input"),
        (circle_axial, "Axial RoPE"),
        (circle_spiral, "Spiral RoPE (ours)"),
    ]

    for axis, (image, title) in zip(axes.ravel(), images):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_masks(axial_mask: np.ndarray, spiral_mask: np.ndarray, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(axial_mask.astype(float))
    axes[0].set_title("Axial mask")
    axes[0].axis("off")

    axes[1].imshow(spiral_mask.astype(float))
    axes[1].set_title("Spiral mask")
    axes[1].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate the paper's FFT-mask-iFFT figure for a point and a circle."
    )
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--rope-dim", type=int, default=1024)
    parser.add_argument("--theta-base", type=float, default=10_000.0)
    parser.add_argument("--num-directions", type=int, default=8)
    parser.add_argument("--circle-radius", type=float, default=15.0)
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("outputs/spiral_rope_fft_reconstruction.png"),
    )
    parser.add_argument(
        "--mask-output",
        type=Path,
        default=Path("outputs/spiral_rope_fft_masks.png"),
    )
    args = parser.parse_args()
    if args.num_directions % 2 != 0:
        parser.error("--num-directions must be even.")
    return args


def main() -> None:
    args = parse_args()
    config = ReconstructionConfig(
        grid_size=args.grid_size,
        rope_dim=args.rope_dim,
        theta_base=args.theta_base,
        num_directions=args.num_directions,
        circle_radius=args.circle_radius,
    )

    point_image = make_point_image(config.grid_size)
    circle_image = make_circle_image(config.grid_size, config.circle_radius)
    theta = rope_frequencies(config)
    wgrid = fft_angular_grid(config.grid_size)

    axial_mask = build_axial_mask(config, theta, wgrid)
    spiral_mask = build_spiral_mask(config, theta, wgrid)

    point_axial = reconstruct(point_image, axial_mask)
    point_spiral = reconstruct(point_image, spiral_mask)
    circle_axial = reconstruct(circle_image, axial_mask)
    circle_spiral = reconstruct(circle_image, spiral_mask)

    plot_reconstructions(
        point_image,
        point_axial,
        point_spiral,
        circle_image,
        circle_axial,
        circle_spiral,
        args.figure_output,
    )
    plot_masks(axial_mask, spiral_mask, args.mask_output)

    print(f"Saved reconstruction figure to {args.figure_output}")
    print(f"Saved mask figure to {args.mask_output}")
    print(f"Axial bins kept: {int(axial_mask.sum())}")
    print(f"Spiral bins kept: {int(spiral_mask.sum())}")
    print(
        "Assumptions: 64x64 centered binary inputs, filled disk radius "
        f"{config.circle_radius}, nearest-bin discretization of the paper's angular-frequency support."
    )


if __name__ == "__main__":
    main()
