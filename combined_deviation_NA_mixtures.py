import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import os

# ====================== INPUT FILES ======================
fiber_file = "photon_exit_data_100MPhotons_reflectance98_IC2_fiber_explicit_setback4_all_batches_deviation_vs_NA.npz"
diffuse_file = "photon_exit_data_100MPhotons_reflectance98_IC2_fiber_explicit_setback4_explicit_tunnel3.2_all_batches_deviation_vs_NA.npz"

# ====================== LOAD DATA ======================
data_fiber = np.load(fiber_file)
data_diffuse = np.load(diffuse_file)

NA_fiber = data_fiber["NA"]
dev_fiber = data_fiber["deviation"]

NA_diffuse = data_diffuse["NA"]
dev_diffuse = data_diffuse["deviation"]

# Safety check: NA grids must match
if not np.allclose(NA_fiber, NA_diffuse):
    raise ValueError("NA grids from the two NPZ files do not match!")

NA = NA_fiber  # common axis

# ====================== MIXTURES ======================
# Pure fiber (0 % diffuse)
dev_0 = dev_fiber

# 5 % diffuse + 95 % fiber
w_diff = 0.05
dev_5 = (1.0 - w_diff) * dev_fiber + w_diff * dev_diffuse

# 10 % diffuse + 90 % fiber
w_diff = 0.10
dev_10 = (1.0 - w_diff) * dev_fiber + w_diff * dev_diffuse

# 20 % diffuse + 80 % fiber
w_diff = 0.20
dev_20 = (1.0 - w_diff) * dev_fiber + w_diff * dev_diffuse

# 50 % diffuse + 50 % fiber
w_diff = 0.50
dev_50 = (1.0 - w_diff) * dev_fiber + w_diff * dev_diffuse

# ====================== GLOBAL STYLE (exact match to analysis script) ======================
plt.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.linewidth': 4,
    'xtick.major.width': 4,
    'ytick.major.width': 4,
    'xtick.major.size': 10,
    'ytick.major.size': 10,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'legend.fontsize': 22,
})

# ====================== FIGURE ======================
fig = plt.figure(figsize=(12, 8))
ax = plt.gca()

# Plot curves (linewidth=4 to match original)
ax.plot(NA, dev_0,  color='purple', linewidth=4, label='fiber (0 % diffuse)')
ax.plot(NA, dev_5,  color='orange', linewidth=4, label='95 % fiber + 5 % diffuse')
ax.plot(NA, dev_10, color='blue',   linewidth=4, label='90 % fiber + 10 % diffuse')
ax.plot(NA, dev_20, color='green',  linewidth=4, label='80 % fiber + 20 % diffuse')
ax.plot(NA, dev_50, color='red',    linewidth=4, label='50 % fiber + 50 % diffuse')

# Horizontal zero line
ax.axhline(0, color='black', linestyle='--', alpha=0.7)

# Axis labels and limits (exact match to Fig. 5B)
ax.set_xlabel(r'Numerical Aperture (NA = $\sin\theta$)')
ax.set_ylabel('Deviation (%)')
ax.set_ylim(-10.5, 10.5)
ax.set_xlim(-0.02, 1.02)

# Major ticks
ax.yaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# Minor ticks
ax.minorticks_on()
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_minor_locator(MultipleLocator(1))

# Tick appearance (exact match)
ax.tick_params(axis='x', which='major', length=12, width=4)
ax.tick_params(axis='x', which='minor', length=10, width=4)
ax.tick_params(axis='y', which='major', length=12, width=4)
ax.tick_params(axis='y', which='minor', length=10, width=4)

# Grid on major + minor
ax.grid(True, which='major', alpha=0.3)
ax.grid(True, which='minor', alpha=0.3)

# Legend
ax.legend(loc='best', frameon=True)

plt.tight_layout()

# ====================== SAVE ======================
out_name = "photon_exit_data_100MPhotons_reflectance98_IC2_fiber_explicit_setback4_diffuse_mixture_deviation_vs_NA.pdf"
plt.savefig(out_name, dpi=300, bbox_inches='tight')
print(f"Figure saved as {out_name}")

plt.show()

