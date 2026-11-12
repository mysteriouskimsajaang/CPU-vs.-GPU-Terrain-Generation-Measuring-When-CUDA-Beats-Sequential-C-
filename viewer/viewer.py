"""Rotatable 3D terrain viewer (blue = low, red = high).

Usage (after running the C++/CUDA program):
    py -m pip install numpy pyvista matplotlib
    py viewer\\viewer.py                         # opens results/hilly_1024x1024.bin
    py viewer\\viewer.py results\\mountains_1024x1024.bin

Drag with the mouse to rotate, scroll to zoom. Not part of any timing.
"""
import sys
from pathlib import Path
import numpy as np

# Default file is found next to this script's repo, so it works from any folder.
ROOT = Path(__file__).resolve().parent.parent
path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results" / "hilly_1024x1024.bin")
MAX_POINTS = 512          # downsample so it stays smooth
HEIGHT_SCALE = 1.0        # raise this (e.g. 2.0) to make hills look taller

with open(path, "rb") as f:
    W, H = np.fromfile(f, dtype=np.int32, count=2)
    h = np.fromfile(f, dtype=np.int32).reshape(H, W).astype(float)

step = max(1, max(W, H) // MAX_POINTS)
h = h[::step, ::step]
print(f"{path}: {W}x{H}, showing every {step} column(s) -> {h.shape[1]}x{h.shape[0]}, "
      f"heights {int(h.min())} to {int(h.max())}")

# x/z spacing in blocks, so the shape is not stretched
x, z = np.meshgrid(np.arange(h.shape[1]) * step, np.arange(h.shape[0]) * step)

try:
    import pyvista as pv
    grid = pv.StructuredGrid(x.astype(float), z.astype(float), h * HEIGHT_SCALE)
    grid["height"] = h.ravel(order="F")
    plotter = pv.Plotter(window_size=(1200, 800))
    plotter.add_mesh(grid, scalars="height", cmap="jet", show_scalar_bar=True)
    plotter.set_scale(zscale=4)   # terrain is ~1000 wide but only ~100 tall
    plotter.add_title(path)
    plotter.show()
except ImportError:
    # Fallback: matplotlib (slower, so downsample more)
    import matplotlib.pyplot as plt
    s = max(1, h.shape[0] // 200)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(x[::s, ::s], z[::s, ::s], h[::s, ::s], cmap="jet",
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.6, label="height (Y)")
    ax.set_box_aspect((1, 1, 0.25))
    ax.set_title(path + "  (install pyvista for a smoother viewer)")
    plt.show()
