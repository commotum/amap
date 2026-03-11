# MonSTERs Spatiotemporal Analysis

## Setup

- Base vectors: 512 random vectors with content dimension 768.
- MonSTER internal dimension: 768 (same as content dimension; no padding required).
- Spatial lattice: 8x8x8 (512 voxels).
- Query t values: -8, -4, 0, 4, 8; key t fixed at 0.
- Families: 512 uniform lattices and 512 random-mix lattices.
- Monte Carlo pairs per family/top-delta combo: 100,000.

## Core Findings

- Absolute-relative fusion holds numerically for both metrics across all tested top_delta values; the sampled errors stayed at machine-precision scale.
- The `(-,+,+,+)` metric is an exact sign flip of `(+,-,-,-)`, so the second metric does not create a new geometry; it just reverses which logits are high versus low.
- Spatial attenuation is the clearest robust trend. Under `(-,+,+,+)`, random-mix lattices consistently give higher logits to space-close pairs than to space-far pairs, and the gap is strongest at smaller top_delta.
- Temporal attenuation is much weaker than spatial attenuation. The mean score versus |Δt| is often flat or nonmonotonic, especially for uniform same-vector lattices, so MonSTERs does not behave like a simple monotone time-decay kernel in this setup.
- Content similarity dominates the mixed-lattice regime: identical or strongly positive-cosine token pairs stay far above neutral or negative-cosine pairs, and this content effect is larger than the average time-sign effect.
- Reducing top_delta strengthens the positional effect, especially in space. The stronger regime (`top_delta=4` or `8`) improves spatial separation, but it still does not turn the temporal behavior into a clean monotone attenuation.

## Absolute-Relative Fusion

- `top_delta=4`: max error `(+,-,-,-)` = 0.000000, max error `(-,+,+,+)` = 0.000000.
- `top_delta=8`: max error `(+,-,-,-)` = 0.000000, max error `(-,+,+,+)` = 0.000000.
- `top_delta=16`: max error `(+,-,-,-)` = 0.000000, max error `(-,+,+,+)` = 0.000000.
- `top_delta=32`: max error `(+,-,-,-)` = 0.000000, max error `(-,+,+,+)` = 0.000000.

## Metric Trend Summary

### (+,-,-,-)

- `uniform`
  - `top_delta=4`: mean=-13.5768, corr(space)=0.1272, corr(|t|)=0.0078, close/close=-13.8550, far/far=-13.3602.
  - `top_delta=8`: mean=-13.7696, corr(space)=0.0359, corr(|t|)=0.0037, close/close=-13.8760, far/far=-13.7228.
  - `top_delta=16`: mean=-13.8325, corr(space)=0.0045, corr(|t|)=0.0031, close/close=-13.9180, far/far=-13.8228.
  - `top_delta=32`: mean=-13.8316, corr(space)=0.0047, corr(|t|)=-0.0049, close/close=-13.8102, far/far=-13.8325.
- `random_mix`
  - `top_delta=4`: mean=-0.0517, corr(space)=0.0498, corr(|t|)=-0.0038, close/close=-0.6432, far/far=-0.0231.
  - `top_delta=8`: mean=-0.0556, corr(space)=0.0574, corr(|t|)=0.0009, close/close=-0.5645, far/far=-0.0132.
  - `top_delta=16`: mean=-0.0476, corr(space)=0.0626, corr(|t|)=-0.0002, close/close=-0.5762, far/far=-0.0170.
  - `top_delta=32`: mean=-0.0549, corr(space)=0.0447, corr(|t|)=0.0013, close/close=-0.3957, far/far=-0.0389.

### (-,+,+,+)

- `uniform`
  - `top_delta=4`: mean=13.5768, corr(space)=-0.1272, corr(|t|)=-0.0078, close/close=13.8550, far/far=13.3602.
  - `top_delta=8`: mean=13.7696, corr(space)=-0.0359, corr(|t|)=-0.0037, close/close=13.8760, far/far=13.7228.
  - `top_delta=16`: mean=13.8325, corr(space)=-0.0045, corr(|t|)=-0.0031, close/close=13.9180, far/far=13.8228.
  - `top_delta=32`: mean=13.8316, corr(space)=-0.0047, corr(|t|)=0.0049, close/close=13.8102, far/far=13.8325.
- `random_mix`
  - `top_delta=4`: mean=0.0517, corr(space)=-0.0498, corr(|t|)=0.0038, close/close=0.6432, far/far=0.0231.
  - `top_delta=8`: mean=0.0556, corr(space)=-0.0574, corr(|t|)=-0.0009, close/close=0.5645, far/far=0.0132.
  - `top_delta=16`: mean=0.0476, corr(space)=-0.0626, corr(|t|)=0.0002, close/close=0.5762, far/far=0.0170.
  - `top_delta=32`: mean=0.0549, corr(space)=-0.0447, corr(|t|)=-0.0013, close/close=0.3957, far/far=0.0389.

## Random-Mix Content Bins

- `top_delta=4`
  - `identical`: n=409, mean=13.7680, close/close=14.1261, far/far=13.6703.
  - `positive_5pct`: n=5036, mean=1.0114, close/close=1.1499, far/far=0.9523.
  - `middle_90pct`: n=89530, mean=-0.0052, close/close=-0.0434, far/far=-0.0057.
  - `negative_5pct`: n=5025, mean=-1.0126, close/close=-0.9629, far/far=-1.0063.
- `top_delta=8`
  - `identical`: n=396, mean=13.7006, close/close=13.6225, far/far=13.0962.
  - `positive_5pct`: n=4990, mean=1.0189, close/close=0.9912, far/far=1.0358.
  - `middle_90pct`: n=89593, mean=0.0005, close/close=0.0031, far/far=-0.0067.
  - `negative_5pct`: n=5021, mean=-0.9961, close/close=-1.1228, far/far=-0.9404.
- `top_delta=16`
  - `identical`: n=382, mean=13.7892, close/close=13.9625, far/far=14.1477.
  - `positive_5pct`: n=4972, mean=1.0054, close/close=1.0548, far/far=0.9970.
  - `middle_90pct`: n=89503, mean=-0.0023, close/close=-0.0521, far/far=-0.0060.
  - `negative_5pct`: n=5143, mean=-1.0294, close/close=-0.9129, far/far=-1.0055.
- `top_delta=32`
  - `identical`: n=398, mean=13.8284, close/close=13.4779, far/far=14.2752.
  - `positive_5pct`: n=5074, mean=1.0262, close/close=1.0142, far/far=1.0484.
  - `middle_90pct`: n=89556, mean=-0.0010, close/close=-0.0111, far/far=0.0068.
  - `negative_5pct`: n=4972, mean=-1.0321, close/close=-0.8874, far/far=-1.0603.

## Time-Sign Effect

- `(+,-,-,-)`
  - `uniform`: `top_delta=4`: Δ(+4,-4)=-0.0171, Δ(+8,-8)=0.0053; `top_delta=8`: Δ(+4,-4)=0.0015, Δ(+8,-8)=0.0117; `top_delta=16`: Δ(+4,-4)=-0.0225, Δ(+8,-8)=-0.0010; `top_delta=32`: Δ(+4,-4)=-0.0106, Δ(+8,-8)=0.0278
  - `random_mix`: `top_delta=4`: Δ(+4,-4)=0.0008, Δ(+8,-8)=0.0327; `top_delta=8`: Δ(+4,-4)=-0.0234, Δ(+8,-8)=0.0162; `top_delta=16`: Δ(+4,-4)=0.0040, Δ(+8,-8)=0.0225; `top_delta=32`: Δ(+4,-4)=0.0118, Δ(+8,-8)=0.0132
- `(-,+,+,+)`
  - `uniform`: `top_delta=4`: Δ(+4,-4)=0.0171, Δ(+8,-8)=-0.0053; `top_delta=8`: Δ(+4,-4)=-0.0015, Δ(+8,-8)=-0.0117; `top_delta=16`: Δ(+4,-4)=0.0225, Δ(+8,-8)=0.0010; `top_delta=32`: Δ(+4,-4)=0.0106, Δ(+8,-8)=-0.0278
  - `random_mix`: `top_delta=4`: Δ(+4,-4)=-0.0008, Δ(+8,-8)=-0.0327; `top_delta=8`: Δ(+4,-4)=0.0234, Δ(+8,-8)=-0.0162; `top_delta=16`: Δ(+4,-4)=-0.0040, Δ(+8,-8)=-0.0225; `top_delta=32`: Δ(+4,-4)=-0.0118, Δ(+8,-8)=-0.0132

## Notes

- The `(-,+,+,+)` metric is exactly the negative of `(+,-,-,-)`, so its logits are sign-flipped versions of the first metric.
- Heatmaps in `monster_spatiotemporal_heatmaps.png` use the `(-,+,+,+)` metric because that is the attenuation-friendly convention.
- Scores here are logits, not post-softmax attention weights.
