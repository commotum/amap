import numpy as np

def tilted_plane_coords(size=32):
    vals = np.arange(-(size/2) + 0.5, size/2, 1.0)  # half-integer centers
    u, v = np.meshgrid(vals, vals, indexing="xy")

    e1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    e2 = np.array([1.0,  1.0, -2.0]) / np.sqrt(6.0)

    x = u * e1[0] + v * e2[0]
    y = u * e1[1] + v * e2[1]
    z = u * e1[2] + v * e2[2]

    return x, y, z