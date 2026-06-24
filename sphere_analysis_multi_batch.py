import numpy as np
import os
import matplotlib.pyplot as plt
import re

# ====================== CONFIG ======================
script_dir = os.path.dirname(os.path.abspath(__file__))

base_name = "photon_exit_data_10MPhotons_reflectance95_fiber_batch"
#"photon_exit_data_10MPhotons_reflectance99_fiber_diameter_port_ratio_3.50_batch"
#"photon_exit_data_10MPhotons_reflectance99_fiber_batch"

# Auto-detect all batches
batch_files = []
i = 1
while True:
    fname = f"{base_name}{i}.npz"
    fpath = os.path.join(script_dir, fname)
    if os.path.exists(fpath):
        batch_files.append(fpath)
        i += 1
    else:
        break

print(f"Found {len(batch_files)} batch file(s) to process.")

if not batch_files:
    print("No batch files found!")
    exit()

# ====================== PORT GEOMETRY ======================
sphere_diameter = 1.75 * 25.4
R_sphere = sphere_diameter / 2

match = re.search(r"port_ratio_([\d\.]+)", base_name)
port_ratio = float(match.group(1)) if match else 2.8
port_diameter = sphere_diameter / port_ratio
r_port = port_diameter / 2

print(f"Using port diameter = {port_diameter:.3f} mm (ratio {port_ratio:.2f})")

alpha_port = np.arcsin(r_port / R_sphere)
port_center = np.array([0, 0, -R_sphere * np.cos(alpha_port)])
port_z = port_center[2]

# ====================== ACCUMULATORS ======================
total_photons = 0
bounces_hist = None
bounces_bin_edges = None
first_bounce_count = 0   # separate counter for exact bounce == 1, this counts the number of photons leaving the sphere after only one bounce 

angle_hist = np.zeros(90, dtype=np.int64)
angle_hist_no1 = np.zeros(90, dtype=np.int64)   # for Figure 4 (>= 2 bounces)
port_2d_hist = None
x_edges_2d = None
y_edges_2d = None

square_angles = [[] for _ in range(9)]

# ====================== PROCESS EACH BATCH ======================
for batch_idx, data_path in enumerate(batch_files, 1):
    print(f"Processing batch {batch_idx}/{len(batch_files)} ...")
    
    data = np.load(data_path)
    old_pos = data['old_positions']
    new_pos = data['new_positions']
    bounces = data['bounces']
    
    n = len(bounces)
    total_photons += n
    
    # Directions and exit angles
    directions = new_pos - old_pos
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / norms
    
    cos_alpha = np.dot(directions, np.array([0., 0., -1.]))
    cos_alpha = np.clip(cos_alpha, -1e-6, 1.0)
    exit_alpha_deg = np.rad2deg(np.arccos(cos_alpha))
    
    # Safety filter
    valid = np.abs(directions[:, 2]) > 1e-9
    removed = n - np.sum(valid)
    if removed > 0:
        print(f"   Removed {removed} grazing rays")
    
    old_pos = old_pos[valid]
    directions = directions[valid]
    bounces = bounces[valid]
    exit_alpha_deg = exit_alpha_deg[valid]
    
    # Port intersections
    t = (port_z - old_pos[:, 2]) / directions[:, 2]
    port_int = old_pos + t[:, np.newaxis] * directions
    
    # Count exact 1-bounce photons
    first_bounce_count += np.sum(bounces == 1)
    
    # Accumulate bounces with custom bins for plotting
    if bounces_hist is None:
        bin_edges_list = np.concatenate([[-4, 0.5], np.arange(5.5, 171, 5)])
        bounces_hist, bounces_bin_edges = np.histogram(bounces, bins=bin_edges_list)
    else:
        bounces_hist += np.histogram(bounces, bins=bounces_bin_edges)[0]
    
    # Accumulate angles
    angle_hist += np.histogram(exit_alpha_deg, bins=90)[0]
    
    # Accumulate angles for photons with >= 2 bounces 
    mask_no1 = bounces >= 2
    if np.any(mask_no1):
        angle_hist_no1 += np.histogram(exit_alpha_deg[mask_no1], bins=90)[0]
    
    # Accumulate 2D heatmap
    if port_2d_hist is None:
        bin_size = 0.5
        x_min = max(np.floor(port_int[:,0].min() - 2), -40)
        x_max = min(np.ceil(port_int[:,0].max() + 2),  40)
        y_min = max(np.floor(port_int[:,1].min() - 2), -40)
        y_max = min(np.ceil(port_int[:,1].max() + 2),  40)
        
        x_edges_2d = np.arange(x_min, x_max + bin_size, bin_size)
        y_edges_2d = np.arange(y_min, y_max + bin_size, bin_size)
        port_2d_hist, _, _ = np.histogram2d(port_int[:,0], port_int[:,1],
                                            bins=[x_edges_2d, y_edges_2d])
    else:
        port_2d_hist += np.histogram2d(port_int[:,0], port_int[:,1],
                                       bins=[x_edges_2d, y_edges_2d])[0]
    
    # Central 3x3 squares
    half = 0.25  # 0.25 = 0.5 / 2 , so half of bin_size
    sq_edges = np.arange(-5.0 - half, 5.0 + half + 0.001, 0.5)
    x_idx = np.digitize(port_int[:,0], sq_edges) - 1
    y_idx = np.digitize(port_int[:,1], sq_edges) - 1
    center_idx = 10
    
    for i in range(3):
        for j in range(3):
            sq = i * 3 + j
            mask = (x_idx == center_idx - 1 + j) & (y_idx == center_idx - 1 + i)
            square_angles[sq].extend(exit_alpha_deg[mask].tolist())

print(f"\nTotal photons processed: {total_photons:,}")

# ====================== GLOBAL STYLE ======================
plt.rcParams.update({
    'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
    'axes.linewidth': 2.5, 'xtick.major.width': 2.5, 'ytick.major.width': 2.5,
    'xtick.major.size': 8, 'ytick.major.size': 8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.labelsize': 16, 'ytick.labelsize': 16,
    'axes.labelsize': 20, 'axes.titlesize': 20, 'legend.fontsize': 14,
})

# ====================== FIGURE 1 ======================
fig1 = plt.figure(1, figsize=(25, 10))

# Left subplot - Bounces
ax1 = plt.subplot(1, 2, 1)
plt.hist(bounces_bin_edges[:-1], bins=bounces_bin_edges, weights=bounces_hist,
         color='skyblue', edgecolor='black')

plt.xlabel('Number of Bounces')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.xlim(-5, 175)

x_labels = np.concatenate([[0], np.arange(5, 171, 5)])
plt.xticks(x_labels, rotation=45, ha='right')

# First-bounce percentage 
percent_first = (first_bounce_count / total_photons) * 100

ax_right = ax1.twinx()
ax_right.stem([1], [percent_first], linefmt='r-', markerfmt='ro', basefmt=' ',
              label=f'1st bounce = {percent_first:.2f}%')
ax_right.set_ylabel('Percentage of Photons (%)', color='red', fontsize=18, labelpad=15)
ax_right.tick_params(axis='y', labelcolor='red')
ax_right.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
ax_right.set_ylim(0, max(10, percent_first * 1.2))
ax_right.legend(loc='upper right', fontsize=14, frameon=True)


# Right subplot - Exit Angle 
plt.subplot(1, 2, 2)

# Plot histogram (left edges + proper bin edges)
plt.hist(np.arange(90), bins=np.arange(91), weights=angle_hist,
         color='salmon', edgecolor='black', alpha=0.75)

# === Statistics ===
values = np.arange(90) + 0.5          # bin centers: 0.5, 1.5, ..., 89.5
cumsum = np.cumsum(angle_hist, dtype=float)
total = cumsum[-1] if cumsum[-1] > 0 else 1.0

mean_alpha   = np.average(values, weights=angle_hist)

right_edges = np.arange(1, 91)             # Important: right edges for CDF
# Better median and 90th percentile using linear interpolation
median_alpha = np.interp(total / 2, cumsum, right_edges)
p90_alpha    = np.interp(0.9 * total, cumsum, right_edges)

plt.axvline(mean_alpha,   color='blue',   linestyle='--', linewidth=4,
            label=f'Mean   = {mean_alpha:.2f}$^\\circ$')
plt.axvline(median_alpha, color='green',  linestyle='-',  linewidth=4,
            label=f'Median = {median_alpha:.2f}$^\\circ$')
plt.axvline(p90_alpha,    color='red',    linestyle=':',  linewidth=4,
            label=f'90th   = {p90_alpha:.2f}$^\\circ$')

plt.xlabel(r'Exit Angle $\theta$ ($^\circ$)')
plt.ylabel('Count')
plt.xticks(np.arange(0, 91, 15))
plt.legend(loc='upper right', fontsize=16)
plt.grid(True, alpha=0.3)

# A/B labels
fig1.text(0.005, 0.95, 'A', fontsize=30, fontweight='bold',
          va='top', ha='left', transform=fig1.transFigure)
fig1.text(0.505, 0.95, 'B', fontsize=30, fontweight='bold',
          va='top', ha='left', transform=fig1.transFigure)

plt.tight_layout(rect=[0, 0, 1, 0.90])


# ====================== FIGURE 2: Heatmap ======================
fig2 = plt.figure(2, figsize=(10, 8))
X, Y = np.meshgrid(x_edges_2d[:-1] + 0.25, y_edges_2d[:-1] + 0.25)
plt.pcolormesh(X, Y, port_2d_hist.T, cmap='viridis', shading='auto')

cbar = plt.colorbar(label='Number of Photons per 0.25 mm$^2$ square')
cbar.ax.tick_params(direction='out')

circle = plt.Circle((0, 0), r_port, fill=False, color='red',
                    linestyle='--', linewidth=4, label='Port Boundary')
plt.gca().add_patch(circle)

plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=16)
plt.tight_layout()

# ====================== FIGURE 3: 3x3 Squares ======================
fig3, axes = plt.subplots(3, 3, figsize=(16, 13), sharex=True, sharey=True)

half = 0.25
sq_edges = np.arange(-5.0 - half, 5.0 + half + 0.001, 0.5)
center_x = np.argmin(np.abs(sq_edges[:-1] + half))

for i in range(3):
    for j in range(3):
        sq_idx = i * 3 + j
        angles = np.array(square_angles[sq_idx])
        ax = axes[i, j]
        
        if len(angles) > 5:
            ax.hist(angles, bins=50, color='salmon', edgecolor='black', alpha=0.75)
            ax.set_xticks(np.arange(0, 91, 15))
            
            m = np.mean(angles)
            med = np.median(angles)
            p90 = np.percentile(angles, 90)
            
            ax.axvline(m, color='blue', linestyle='--', linewidth=3,
                       label=f'Mean = {m:.2f}$^\\circ$')
            ax.axvline(med, color='green', linestyle='-', linewidth=3,
                       label=f'Median = {med:.2f}$^\\circ$')
            ax.axvline(p90, color='red', linestyle=':', linewidth=3,
                       label=f'90th = {p90:.2f}$^\\circ$')
            
            ax.legend(loc='lower left', bbox_to_anchor=(0.03, 0.03), fontsize=11)
            ax.set_title(f'Square ({j-1:+d}, {i-1:+d})\nN = {len(angles)}', fontsize=11)
        else:
            ax.text(0.5, 0.5, 'Too few photons\nfor statistics',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(f'Square ({j-1:+d}, {i-1:+d})', fontsize=11)
        
        ax.set_xlabel(r'Exit Angle $\theta$ ($^\circ$)')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# ====================== FIGURE 4: Exit Angles without 1st Bounce ======================
fig4 = plt.figure(4, figsize=(12, 8))

# Plot histogram correctly (left edges + proper bin edges)
plt.hist(np.arange(90), bins=np.arange(91), weights=angle_hist_no1,
         color='salmon', edgecolor='black', alpha=0.75)

# === Statistics ===
values = np.arange(90) + 0.5          # bin centers: 0.5, 1.5, ..., 89.5
cumsum = np.cumsum(angle_hist_no1, dtype=float)
total = cumsum[-1] if cumsum[-1] > 0 else 1.0

mean_alpha = np.average(values, weights=angle_hist_no1)

right_edges = np.arange(1, 91)             # Important: right edges for CDF
# Better median and 90th percentile using linear interpolation
median_alpha = np.interp(total / 2, cumsum, right_edges)
p90_alpha    = np.interp(0.9 * total, cumsum, right_edges)

plt.axvline(mean_alpha,   color='blue',   linestyle='--', linewidth=4, label=f'Mean   = {mean_alpha:.2f}$^\\circ$')
plt.axvline(median_alpha, color='green',  linestyle='-',  linewidth=4, label=f'Median = {median_alpha:.2f}$^\\circ$')
plt.axvline(p90_alpha,    color='red',    linestyle=':',  linewidth=4, label=f'90th   = {p90_alpha:.2f}$^\\circ$')

plt.xlabel(r'Exit Angle $\theta$ ($^\circ$) - Photons with $\geq$ 2 bounces')
plt.ylabel('Count')
plt.xticks(np.arange(0, 91, 15))
plt.legend(loc='upper right', fontsize=16)
plt.grid(True, alpha=0.3)

plt.tight_layout()

# ====================== FIGURE 5: Central Square CDF vs Lambertian ======================
fig5 = plt.figure(5, figsize=(14, 7))

# Find central square (0,0) with index 4
central_angles = np.array(square_angles[4])

if len(central_angles) > 10:
    # Histogram
    hist_central, bin_edges = np.histogram(central_angles, bins=90, range=(0, 90))
    
    # Cumulative sum to normalized CDF
    cdf_central = np.cumsum(hist_central, dtype=float)
    cdf_central /= cdf_central[-1]

    # Use RIGHT bin edges for x-axis (this is the correct alignment)
    theta_right = bin_edges[1:]                    # 1, 2, ..., 90 degrees 
    theta_rad = np.deg2rad(theta_right)
    ideal_cdf = np.sin(theta_rad)**2

    # === Subplot A: CDF comparison ===
    axA = plt.subplot(1, 2, 1)
    axA.plot(theta_right, cdf_central, 'b-', linewidth=3, label='Measured CDF (central square)')
    axA.plot(theta_right, ideal_cdf, 'r--', linewidth=2.5, label=r'Ideal Lambertian ($\sin^2\theta$)')
    
    axA.set_xlabel(r'Exit Angle $\theta$ ($^\circ$)')
    axA.set_ylabel('Cumulative Fraction')
    axA.set_ylim(0, 1.02)
    axA.set_xlim(0, 90)
    axA.set_xticks(np.arange(0, 91, 15))
    axA.grid(True, alpha=0.3)
    axA.legend(loc='lower right', fontsize=14)

    # === Subplot B: Deviation vs NA ===
    axB = plt.subplot(1, 2, 2)
    deviation = 100 * (cdf_central - ideal_cdf) / (ideal_cdf + 1e-12)
    
    NA = np.sin(theta_rad)
    
    axB.plot(NA, deviation, 'purple', linewidth=3)
    axB.axhline(0, color='black', linestyle='--', alpha=0.7)
    
    axB.set_xlabel(r'Numerical Aperture (NA = $\sin\theta$)')
    axB.set_ylabel('Deviation from Lambertian (%)')
    axB.set_ylim(-10.5, 0.5)
    axB.set_yticks(np.arange(-10, 1, 1))
    axB.set_xlim(0.0, 1.02)
    axB.set_xticks(np.arange(0.0, 1.1, 0.1))
    axB.grid(True, alpha=0.3)

    # Large A/B labels (matching Figure 1 style)
    fig5.text(0.005, 0.95, 'A', fontsize=30, fontweight='bold',
              va='top', ha='left', transform=fig5.transFigure)
    fig5.text(0.505, 0.95, 'B', fontsize=30, fontweight='bold',
              va='top', ha='left', transform=fig5.transFigure)

else:
    print("Warning: Not enough photons in central square for Figure 5")
    fig5.text(0.5, 0.5, 'Not enough photons in central square', 
              ha='center', va='center', fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.93])

# ====================== FIGURE 6: Diagonal Uniformity ======================
fig6 = plt.figure(6, figsize=(12, 7))

bin_size = 0.5
x_centers = (x_edges_2d[:-1] + x_edges_2d[1:]) / 2
y_centers = (y_edges_2d[:-1] + y_edges_2d[1:]) / 2

# fully inside bins (4 corners check)
inside_mask = np.zeros(port_2d_hist.shape, dtype=bool)

for i in range(len(x_centers)):
    for j in range(len(y_centers)):
        # Define the four corners of this bin
        corners = [
            (x_edges_2d[i],   y_edges_2d[j]),     # bottom-left
            (x_edges_2d[i+1], y_edges_2d[j]),     # bottom-right
            (x_edges_2d[i],   y_edges_2d[j+1]),   # top-left
            (x_edges_2d[i+1], y_edges_2d[j+1])    # top-right
        ]
        
        # Bin is fully inside only if ALL four corners are inside the port
        all_inside = all(np.sqrt(cx**2 + cy**2) <= r_port for cx, cy in corners)
        inside_mask[i, j] = all_inside

# Calculate average photon count using only fully-inside bins
inside_counts = port_2d_hist[inside_mask]
average_photons = np.mean(inside_counts) if np.sum(inside_mask) > 0 else 1.0

print(f"Number of fully inside bins (all 4 corners inside): {np.sum(inside_mask)}")
print(f"Average photons per inside bin: {average_photons:.2f}")

# extract diagonals 
diag1_pos = []
diag1_percent = []
diag2_pos = []
diag2_percent = []

for i in range(len(x_centers)):
    for j in range(len(y_centers)):
        r = np.sqrt(x_centers[i]**2 + y_centers[j]**2)
        if r > r_port:
            continue
            
        count = port_2d_hist[i, j]
        percent = (count / average_photons) * 100
        
        # Diagonal 1: x = y  along injection direction [1,1,1]
        if abs(x_centers[i] - y_centers[j]) < bin_size / 2:
            pos = x_centers[i] * np.sqrt(2)          # signed distance along diagonal
            diag1_pos.append(pos)
            diag1_percent.append(percent)
        
        # Diagonal 2: x = -y  orthogonal to injection direction
        if abs(x_centers[i] + y_centers[j]) < bin_size / 2:
            pos = x_centers[i] * np.sqrt(2)
            diag2_pos.append(pos)
            diag2_percent.append(percent)

# sort diagonals 
# Diagonal 1: x = y
sort_idx = np.argsort(diag1_pos)
diag1_pos = np.array(diag1_pos)[sort_idx]
diag1_percent = np.array(diag1_percent)[sort_idx]

# Diagonal 2: x = -y
sort_idx = np.argsort(diag2_pos)
diag2_pos = np.array(diag2_pos)[sort_idx]
diag2_percent = np.array(diag2_percent)[sort_idx]

# plot 
plt.plot(diag1_pos, diag1_percent, 'o-', color='blue', linewidth=3, markersize=7,
         label='Diagonal 1: x = y (along [1,1,1] injection)')
plt.plot(diag2_pos, diag2_percent, 'o-', color='red', linewidth=3, markersize=7,
         label='Diagonal 2: x = -y (orthogonal)')

plt.axhline(100, color='black', linestyle='--', linewidth=2, label='100% (average)')

plt.xlabel('Position along diagonal (mm)')
plt.ylabel('Photon count (% of average inside port)')
plt.xlim(-r_port * 1.05, r_port * 1.05)
plt.ylim(95, 105)                    # good range for seeing non-uniformity
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=14)

plt.tight_layout()

# ====================== SAVE FIGURES ======================
save_base = base_name.replace("batch", "all_batches") if len(batch_files) > 1 else os.path.basename(batch_files[0]).replace('.npz', '')
save_base = save_base.replace("10MPhotons",f"{len(batch_files)*10}MPhotons") 

plt.figure(1).savefig(f"{save_base}_bounces_and_angles.pdf", dpi=300, bbox_inches='tight')
plt.figure(2).savefig(f"{save_base}_port_heatmap.pdf", dpi=300, bbox_inches='tight')
plt.figure(3).savefig(f"{save_base}_3x3_angle_distributions.pdf", dpi=300, bbox_inches='tight')
plt.figure(4).savefig(f"{save_base}_angles_above_1bounce.pdf", dpi=300, bbox_inches='tight')
plt.figure(5).savefig(f"{save_base}_central_square_CDF_vs_Lambertian.pdf", dpi=300, bbox_inches='tight')
plt.figure(6).savefig(f"{save_base}_diagonal_uniformity.pdf", dpi=300, bbox_inches='tight')

print("\nAll figures saved successfully as PDFs!")
plt.show()