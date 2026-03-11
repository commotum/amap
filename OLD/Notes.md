# `rope_attention_single_color.py` Notes

## Bugs or Issues to Fix

- Incomplete CLI validation: [`parse_args()`](/home/jake/Developer/amap/rope_attention_single_color.py#L214) checks `embed_dim % 4 == 0` and `embed_dim % num_directions == 0`, but Spiral RoPE also requires `(embed_dim // num_directions) % 2 == 0`. Right now some invalid configs pass argument parsing and only fail later inside `spiral_frequency_sets()`.
- Unused argument: [`spiral_encode()`](/home/jake/Developer/amap/rope_attention_single_color.py#L96) accepts `theta_base` but does not use it. The frequencies are already precomputed before that function is called.
- Dead code: [`embedding_to_color()`](/home/jake/Developer/amap/rope_attention_single_color.py#L157) is unused.

## Overall Purpose

The script is a visualization tool for comparing how two positional encoding schemes, Axial RoPE and Spiral RoPE, shape attention on a 2D grid when the content is held constant everywhere.

Its real goal is not image understanding. Instead, it isolates positional effects by giving every pixel the same base embedding, selecting one query pixel, and then plotting the attention logits induced purely by position encoding.

## Processing Flow

1. Parse command-line arguments such as image size, embedding dimensions, query location, and RoPE hyperparameters.
2. Create one normalized random embedding vector that will be reused at every pixel.
3. Enumerate all `(x, y)` positions in the image grid.
4. Encode the query and every key position with Axial RoPE.
5. Encode the same query and keys with Spiral RoPE.
6. Compute dot-product attention logits from the query to every position for each encoding scheme.
7. Build a simple binary input image with a single active query pixel for reference.
8. Render three panels: the binary input, the Axial RoPE attention map, and the Spiral RoPE attention map.
9. Save one output image per requested embedding dimension.

## Practical Interpretation

- The plotted heatmaps show positional bias, not learned semantic attention.
- The input image is only a visual marker for the query location.
- Because queries and keys share the same base embedding everywhere, any structure in the heatmaps comes from the positional encoding alone.

## Performance Notes After Reviewing `MonSTERs/v12.py`

### Correction

The MonSTER implementation does not replace the main score computation with a relative-position shortcut.

In [`MonSTERs/v12.py`](/home/jake/Developer/amap/MonSTERs/v12.py#L157), it first computes the full absolute-position transforms for both query and key and scores those transformed vectors:

- `T_abs_q`, `T_abs_k`
- `q_abs`, `k_abs`
- `lhs = minkowski_dot_big_vec(q_abs, k_abs)`

Only after that does it compute the relative-position version:

- `T_rel`
- `k_rel`
- `rhs = minkowski_dot_big_vec(q, k_rel)`

That second path is there to verify the absolute-relative fusion identity, not to replace the primary computation.

### Exact Performance Improvements Worth Porting

These improvements preserve the current semantics of `rope_attention_single_color.py`. They do not depend on scoring directly from relative offsets.

1. Cache invariant frequency tables once per configuration.

- [`axial_encode()`](/home/jake/Developer/amap/rope_attention_single_color.py#L51) recomputes `base_frequencies()` on every call.
- [`spiral_frequency_sets()`](/home/jake/Developer/amap/rope_attention_single_color.py#L67) is rebuilt inside the attention-map computation path.
- MonSTER moves this kind of setup into initialization in [`TriadMonSTERFastVec.__init__()`](/home/jake/Developer/amap/MonSTERs/v12.py#L40).

2. Batch the position encoding across all pixels.

- [`compute_attention_map()`](/home/jake/Developer/amap/rope_attention_single_color.py#L119) currently loops over positions in Python and encodes one key at a time.
- The MonSTER speed pattern is to reshape once and apply broadcasted operations over whole blocks in [`apply_monster_triad_fast_vec()`](/home/jake/Developer/amap/MonSTERs/v12.py#L81).
- Axial and Spiral RoPE should follow that same pattern: one batched transform over all positions, not one Python call per pixel.

3. Precompute trig tables for every grid position.

- For Axial RoPE, precompute `cos(x * freqs)`, `sin(x * freqs)`, `cos(y * freqs)`, and `sin(y * freqs)` for all x/y coordinates in the grid.
- For Spiral RoPE, precompute direction angles and projected coordinates for all positions, then build the corresponding trig tables once.
- This matches the MonSTER idea of precomputing scalar tables in [`TriadMonSTERFastVec.forward()`](/home/jake/Developer/amap/MonSTERs/v12.py#L51), except here the tables are Euclidean rotation tables instead of Lorentz scalar tables.

4. Operate on reshaped pair blocks instead of Python lists.

- [`spiral_encode()`](/home/jake/Developer/amap/rope_attention_single_color.py#L96) builds `encoded_groups` in a Python loop and concatenates them at the end.
- A faster exact implementation would reshape the embedding once into grouped 2D pairs and rotate the whole tensor with broadcasting.
- This is the same structural idea as MonSTER reshaping into block form before applying vectorized updates.

5. Cache reshaped base-vector views.

- The same base embedding is reused for every position in `rope_attention_single_color.py`.
- That means the split/reshape operations for axial halves and spiral groups should also be computed once and reused.
- MonSTER similarly avoids repeated structural work by keeping a fixed block layout throughout the transform.

6. Use dense position arrays instead of `list[tuple[int, int]]`.

- The current code builds a Python list of `(x, y)` tuples and iterates over it.
- A NumPy array of positions would make batched projections and batched rotations straightforward and remove Python-loop overhead.

### Practical Summary

The useful MonSTER lesson is not "switch to relative scoring." The useful lesson is:

- precompute invariant tables once
- keep the embedding in a block/pair layout
- apply the transform with broadcasting over all blocks and positions
- avoid Python loops in the hot path

That gives a real performance improvement while preserving the exact same full absolute-position computation that the current Axial and Spiral RoPE visualizer is trying to show.
