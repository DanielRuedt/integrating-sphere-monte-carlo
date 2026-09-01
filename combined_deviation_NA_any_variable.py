import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter 

# ====================== GLOBAL STYLE ======================
plt.rcParams.update({
    'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
    'axes.linewidth': 4, 'xtick.major.width': 4, 'ytick.major.width': 4,
    'xtick.major.size': 10, 'ytick.major.size': 10,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.labelsize': 20, 'ytick.labelsize': 20,
    'axes.labelsize': 24, 'axes.titlesize': 24, 'legend.fontsize': 22,
})

# ====================== USER CONFIGURATION ======================
# === 1. List your data files (full paths or filenames if in same folder) ===
data_files = [
    "photon_exit_data_100MPhotons_reflectance99_equatorial_fiber_diameter_port_ratio_3.50_all_batches_deviation_vs_NA.npz",  # index 0
    "photon_exit_data_100MPhotons_reflectance99_equatorial_fiber_largeNA_diameter_port_ratio_3.50_all_batches_deviation_vs_NA.npz",  # index 1
    #"photon_exit_data_100MPhotons_reflectance97_equatorial_fiber_offset-15_all_batches_deviation_vs_NA.npz",  # index 2
    # Add/remove files as needed
]

# === 2. Corresponding values of your variable (must be same length as data_files) === 
# Example: if your variable is wall reflectance in %  [97, 98, 99]
# Example: if your variable is offset in degrees in [-15, 0, +15]
# Example: if your variable is fiber NA in um  [0.22, 0.39]
variable_values = [0.22, 0.39]          # <<< CHANGE THIS
variable_name = r"Injection NA (no unit)"   # <<< Label for x-axis in B and C, e.g. r"Injection Angle ($^\circ$)"
variable_unit = r""                    # <<< Used in legends if needed, e.g. r"$^\circ$"

save_name = "photon_exit_data_100MPhotons_reflectance99_equatorial_fiber_diameter_port_ratio_3.50_injection_NAs_combined"  # This is the save name that will be used 

# === 3. Objectives (fixed points for subplot B) ===
# Change NA values and labels as you wish 
objectives = [
    {"NA": 0.40, "label": "NA 0.40", "color": "#1f77b4"},
    {"NA": 0.65, "label": "NA 0.65", "color": "#ff7f0e"},
    {"NA": 0.90, "label": "NA 0.90", "color": "#2ca02c"},
]

# ====================== LOAD DATA ======================
if len(data_files) != len(variable_values):
    raise ValueError("data_files and variable_values must have the same length!")

data_dict = {}
central_counts = {}
loaded_values = []

for i, fpath in enumerate(data_files):
    if not os.path.exists(fpath):
        print(f"File not found: {fpath}")
        continue
    
    val = variable_values[i]
    loaded_values.append(val)
    
    data = np.load(fpath)
    data_dict[val] = data
    central_counts[val] = data['central_square_count']

# Sort by variable value
loaded_values = sorted(loaded_values)

print(f"Loaded {len(loaded_values)} datasets for variable values: {loaded_values}")

# ====================== FIGURE 1: Main Plot (A + B) ======================
fig1, (axA, axB) = plt.subplots(1, 2, figsize=(18, 7.5))

# Subplot A: Deviation vs NA for each variable value
colors_A = plt.cm.viridis(np.linspace(0.15, 0.85, len(loaded_values)))
for val, color in zip(loaded_values, colors_A):
    d = data_dict[val]
    axA.plot(d['NA'], d['deviation'], linewidth=4, color=color,
             label=f'{val}{variable_unit}')

axA.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.7)
axA.set_xlabel(r'Numerical Aperture (NA = $\sin\theta$)')
axA.set_ylabel('Deviation from Lambertian (%)')
axA.set_ylim(-10.5, 10.5)
axA.set_xlim(-0.02, 1.02)
axA.legend(loc='lower right', fontsize=22, frameon=True)

# Major ticks
axA.yaxis.set_major_locator(MultipleLocator(2))
axA.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
axA.xaxis.set_major_locator(MultipleLocator(0.1))
axA.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# Minor ticks
axA.minorticks_on()
axA.xaxis.set_minor_locator(MultipleLocator(0.05))
axA.yaxis.set_minor_locator(MultipleLocator(1))

# Tick appearance (exact match)
axA.tick_params(axis='x', which='major', length=12, width=4)
axA.tick_params(axis='x', which='minor', length=10, width=4)
axA.tick_params(axis='y', which='major', length=12, width=4)
axA.tick_params(axis='y', which='minor', length=10, width=4)

# Grid on major + minor
axA.grid(True, which='major', alpha=0.3)
axA.grid(True, which='minor', alpha=0.3)

fig1.text(0.005, 0.97, 'A', fontsize=30, fontweight='bold', va='top', ha='left', transform=fig1.transFigure)

# Subplot B: Deviation at fixed NAs vs your variable
for obj in objectives:
    dev_at_NA = [np.interp(obj["NA"], data_dict[r]['NA'], data_dict[r]['deviation']) 
                 for r in loaded_values]
    axB.plot(loaded_values, dev_at_NA, 'o-', linewidth=4, markersize=8,
             color=obj["color"], label=obj["label"])

axB.set_xlabel(variable_name)
axB.set_ylabel('Deviation from Lambertian (%)')
axB.set_ylim(-10.5, 10.5)
axB.legend(loc='upper right', fontsize=22, frameon=True)

# Major ticks
axB.yaxis.set_major_locator(MultipleLocator(2))
axB.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
axB.xaxis.set_major_locator(MultipleLocator(0.5))
axB.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# Minor ticks
axB.minorticks_on()
axB.xaxis.set_minor_locator(MultipleLocator(0.1))
axB.yaxis.set_minor_locator(MultipleLocator(1))

# axB.set_xticks(loaded_values)

# Tick appearance (exact match)
axB.tick_params(axis='x', which='major', length=12, width=4)
axB.tick_params(axis='x', which='minor', length=10, width=4)
axB.tick_params(axis='y', which='major', length=12, width=4)
axB.tick_params(axis='y', which='minor', length=10, width=4)

# Grid on major + minor
axB.grid(True, which='major', alpha=0.3)
axB.grid(True, which='minor', alpha=0.3)

# Optional: adjust x-ticks if your variable values are not evenly spaced
# axB.set_xticks(loaded_values)

fig1.text(0.505, 0.97, 'B', fontsize=30, fontweight='bold', va='top', ha='left', transform=fig1.transFigure)

plt.tight_layout(rect=[0, 0, 1, 0.93])

fig1.savefig(f"{save_name}_combined_plot.pdf", dpi=300, bbox_inches='tight')
print(f"Figure 1 saved as: {save_name}_combined_plot.pdf")

# ====================== FIGURE 3: Normalized Central Square Counts ======================
fig3 = plt.figure(figsize=(9, 7))
ax3 = fig3.add_subplot(111)

counts = [central_counts[v] for v in loaded_values]
max_count = max(counts)
norm_percent = [100 * c / max_count for c in counts]

ax3.plot(loaded_values, norm_percent, 'o-', linewidth=3.5, markersize=9,
         color='#d62728', label='Central square (0,0)')

ax3.set_xlabel(variable_name)
ax3.set_xticks(variable_values)
ax3.set_ylabel('Central Square Photon Count\n(% of maximum)')
ax3.set_ylim(0, 110)
ax3.grid(True, alpha=0.3)

# Uncomment to show values on points
# for x, y in zip(loaded_values, norm_percent):
#     ax3.text(x, y + 0.4, f'{y:.1f}%', ha='center', va='bottom', 
#              fontsize=13, fontweight='bold')

ax3.legend(loc='lower right', fontsize=14, frameon=True)

fig3.text(0.02, 0.94, 'C', fontsize=30, fontweight='bold',
          va='top', ha='left', transform=fig3.transFigure)

plt.tight_layout()

fig3.savefig(f"{save_name}_combined_plot_CentralSquareCounts.pdf", dpi=300, bbox_inches='tight')
print(f"Figure 3 saved as: {save_name}_combined_plot_CentralSquareCounts.pdf")

plt.show()

