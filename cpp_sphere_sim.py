import numpy as np
import sphere_sim_cpp
import os
import time

# This code simulates the photon flux generated at the port of an integrating sphere. It assumes perfect lambertian reflection on the inner sphere 
# surface. The reflectance is modelled explicitly by the reflectance value and the probability of absorption at every bounce. The probability of absorption is 1 - P(reflection). 
# For the reflection, the polar angle relative to the surface normal and the azimuthal angle are calculated using the method described in 
# Prokhorov et al. 2003, "Monte Carlo modeling of an integrating sphere reflectometer" DOI: 10.1364/AO.42.003832 , https://www.virial.com/pdf/AO2003-42-19-3832-3842_IS.pdf 
# The unit of length used for this simulation is mm, i.e. milli meters. 


# ====================== BEGINNING SIMULATION PARAMETERS ======================
bounces = 460               # Maximum bounces before terminating a photon.
                            # Chosen so that reflectance^n < 0.01 (i.e. >99% absorbed on average)
                            # Values: 460 for reflectance=0.99, 90 for 0.95, 44 for 0.90, 
                            # 29 for 0.85, 21 for 0.80 
reflectance = 0.98          

batches = 10
photons = 10 * (10**6)      # photons per batch

sphere_diameter = 1.75 * 25.4
R_sphere = sphere_diameter / 2

port_diameter = (5/8) * 25.4        # port diameter in mm. (5/8) * 25.4 is the standard for the IC2 integrating sphere from StellarNet 
r_port = port_diameter / 2

origin = np.array([0, 0, 0])        # the center of the sphere is at the origin 

alpha_port = np.arcsin(r_port / R_sphere)
port_center = np.array([0, 0, -R_sphere * np.cos(alpha_port)])

injection_location = "IC2"                       # "equatorial" = [1,1,0] direction , "polar" = [0,0,1] direction, "111direction" = [1,1,1], IC2 
# For "IC2": 28 degrees downward angle and 30 degrees relative to x-axis for the injection direction, injection_point = P1 = [0.334433 0.615735 0.524071] (approximately in inches)
# injection from the [1,1,1] direction [R_sphere/np.sqrt(3), R_sphere/np.sqrt(3), R_sphere/np.sqrt(3)], 
# equatorial injection [1,1,0] [R_sphere/np.sqrt(2), R_sphere/np.sqrt(2), 0] 
injection_point = np.array([R_sphere/np.sqrt(2), R_sphere/np.sqrt(2), 0]) 

explicit_direction = False  #If you want to use an explicit injection direction, set explicit_direction = True and give the explicit_injection_direction as a numpy array 
                            #The injection cone, i.e. the injection directions of the injected photons, then has its axis along the explicit_injection_direction. 
explicit_injection_direction = None
explicit_setback = 0        # If you want to move the origin of the emission outside the sphere, use a value > 0, such as 4 for 4 mm setback injection (approximately for the IC2)
                            # This only works if explicit_direction = True and an explicit_injection_direction is given. 
                            # When explicit_setback is active, it is assumed that all photons injected in this way actually reach the far side of the sphere, or exit the sphere directly
                            # through the port opening. Use an explicit_injection_direction and an explicit_setback that ensures this is the case to avoid photons entirely missing 
                            # the sphere. 
                            # This can be used to simulate directional injection through a tunnel. The simulation assumes that no photons exit the sphere through the injection tunnel. 
                            # The only way that photons can exit the sphere is through the port opening at the bottom. 
explicit_disc = 1.5         # Beam homogenization and ray mixing is expected in the injection tunnel of the sphere due to specular reflection on the inner surface of the SMA receptacle. 
                            # Gives the beam radius in mm, angular distribution is unaffected. 
                            # explicit_disc applies to explicit_direction = True and explicit_setback has to be > 0 and large enough in order for explicit_disc to work. 
                            # for the IC2, the light first goes through the inner diameter of the SMA receptacle, which is 3.2 mm wide and the setback is about 4 mm.
                            # The photons are then injected from a disc shaped region outside of the sphere that is perpendicular to the explicit_injection_direction and 
                            # centered on the axis created by the injection_point and the explicit_injection_direction. 
explicit_tunnel = 0         # for explicit_direction = True, specifies a tunnel radius for the injection tunnel where diffuse reflection occurs. The tunnel length in mm along the 
                            # central tunnel axis is equal to explicit_setback and for the IC2, the tunnel radius is approximately 3.2 mm. 
                            # If explicit_tunnel = 0, then diffuse reflection at the tunnel is ignored. 
                            # If explicit_tunnel > 0 then the injection is simulated as entirely coming from diffuse reflection at the tunnel wall, so injection type is ignored. 
                            # Wall points are drawn randomly from the cylindrical surface on an axial interval up to 3 * explicit_setback, so the curved cylinder-sphere cut is included; 
                            # points that lie inside the sphere are then rejected. This ensures that the sampled wall points extend all the way to the sphere surface. 
                            # Be careful not to choose a too large explicit_setback, otherwise the cylinder can reach the opposite side 
                            # of the sphere, which renders the rejection invalid. 
                            # Only photons with injection directions that actually pass through the tunnel are simulated. 
                            # This models the tunnel wall as a uniform diffuse radiator. 

if injection_location == "IC2": 
    # ----- IC2 injection geometry ----- 
    port_center_z  = port_center[2]

    # Starting point outside the sphere
    P0 = np.array([1.0 * 25.4, 1.0 * 25.4, port_center_z + 1.75 * 25.4])

    # Direction (measured as 28 degrees down, 30 degrees relative to x axis ) 
    theta_down = np.deg2rad(28)
    phi_x        = np.deg2rad(30)

    horiz = np.cos(theta_down)
    D = np.array([
        horiz * np.cos(phi_x),
        horiz * np.sin(phi_x),
        np.sin(theta_down)
    ])
    D = -D                          # point inward & downward
    D = D / np.linalg.norm(D)       # unit vector

    # Ray/sphere intersection 
    A = np.dot(D, D)
    B = 2 * np.dot(D, P0)
    C = np.dot(P0, P0) - R_sphere**2
    disc = B**2 - 4*A*C

    if disc < 0:
        raise ValueError("No real intersection for IC2 injection ray")

    sqrt_disc = np.sqrt(disc)
    t1 = (-B - sqrt_disc) / (2*A)   # near intersection
    P1 = P0 + t1 * D

    # Set the actual injection point used by the Monte Carlo
    injection_point = P1         
    explicit_direction = True 
    explicit_injection_direction = D.copy()
    explicit_setback = 4        # setback estimated for the IC2 in mm is 4 mm 

injection_type = "fiber"                  # "laser", "fiber", "diffuse", "fiber_largeNA", "..._offsetX" (X > 0 more downward injection in degrees, X < 0 more upward injection) 
                                          # "fiber" -> fiber optic cable with NA 0.22, "fiber_largeNA" -> fiber optic cable with NA 0.39 

if injection_type == "diffuse": 
    explicit_direction = False             # when using diffuse injection you cannot use an explicit_direction, but you can still use an injection point that is calculated based on 
                                           # the tunnel geometry 
    explicit_injection_direction = None
    explicit_setback = 0                   # using diffuse injection while at the same time moving the origin of emission outside the sphere produces invalid results 
    explicit_disc = 0                      # setting explicit_disc > 0 can lead to the origin of emission moving outside the sphere, so with diffuse injection, it is set to 0 

if explicit_tunnel > 0: 
    explicit_disc = 0                      # when using explicit tunnel to simulate diffuse injection via the tunnel, then you cannot use an explicit disk 

# ===================== END SIMULATION PARAMETERS ====================== 

# ------------------------------------------------------------------
# Sanity check: ensure a real tunnel exists (outer cross-section
# lies entirely outside the sphere) if explicit_tunnel is > 0 
if explicit_direction and explicit_injection_direction is not None and explicit_setback > 0 and explicit_tunnel > 0:
    setback_point = injection_point - explicit_setback * explicit_injection_direction

    # Build orthonormal basis perpendicular to the tunnel axis
    axis = explicit_injection_direction / np.linalg.norm(explicit_injection_direction)
    arbitrary = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, arbitrary)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)

    n_samples = 64
    for i in range(n_samples):
        theta = 2.0 * np.pi * i / n_samples
        pt = (setback_point
              + explicit_tunnel * (np.cos(theta) * e1 + np.sin(theta) * e2))
        if np.linalg.norm(pt) <= R_sphere:
            raise RuntimeError(
                "No valid tunnel exists for the given geometry.\n"
                f"  injection_point          = {injection_point}\n"
                f"  explicit_injection_dir   = {explicit_injection_direction}\n"
                f"  explicit_setback         = {explicit_setback} mm\n"
                f"  explicit_tunnel (radius) = {explicit_tunnel} mm\n\n"
                "At least one point on the outer circular cross-section of the "
                "tunnel lies inside or on the sphere surface. This means the "
                "tunnel walls do not extend all the way around.\n\n"
                "Possible remedies:\n"
                "  -> increase explicit_setback, or\n"
                "  -> reduce explicit_tunnel (the tunnel radius)."
            )
# ------------------------------------------------------------------


print("Simulating photon flux at the port of an integrating sphere")
print(f"Reflectance = {reflectance*100:.1f}% | Total photons = {batches * photons:,}\n")

script_dir = os.path.dirname(os.path.abspath(__file__))

for batch in range(1, batches + 1):
    print(f"Batch {batch}/{batches} - Simulating {photons:,} photons...")

    start_time = time.time()

    # ==================== CALL C++ MODULE ====================
    old_pos, new_pos, bounce_numbers = sphere_sim_cpp.simulate_photons(
        n_photons=photons,
        R_sphere=R_sphere,
        max_bounces=bounces,
        reflectance=reflectance,
        injection_point=injection_point,
        injection_type=injection_type,
        port_center=port_center,
        explicit_direction=explicit_direction,
        explicit_injection_direction=explicit_injection_direction,
        explicit_setback=explicit_setback,
        explicit_disc=explicit_disc,
        explicit_tunnel=explicit_tunnel
    )
    # ========================================================

    elapsed = time.time() - start_time

    exit_count = len(bounce_numbers)

    print(f"Finished in {elapsed:.2f} seconds")
    print(f"Photons exited through port: {exit_count:,} / {photons:,} "
          f"({exit_count/photons*100:.2f}%)")

    # Optional: print first 10 old positions (like in your original script)
    if exit_count > 0:
        print("First 5 old positions:")
        for i in range(min(5, exit_count)):
            print(f"     {old_pos[i]}")

    # ====================== SAVE RESULTS ======================
    sim_photon_number_millions = photons // 1_000_000

    # Build filename with injection location clearly marked
    base_name = (f"photon_exit_data_{sim_photon_number_millions}MPhotons_"
                 f"reflectance{int(reflectance*100)}_"
                 f"{injection_location}_{injection_type}")

    if explicit_disc > 0:
        base_name += f"_explicit_disc{explicit_disc}"

    if explicit_setback > 0: 
        base_name += f"_explicit_setback{explicit_setback}"

    if explicit_tunnel > 0:
        base_name += f"_explicit_tunnel{explicit_tunnel}"

    if abs(sphere_diameter / port_diameter - 1.75 / (5/8)) < 1e-6:
        save_path = os.path.join(script_dir, f"{base_name}_batch{batch}.npz")
    else:
        save_path = os.path.join(script_dir, 
            f"{base_name}_diameter_port_ratio_{sphere_diameter/port_diameter:.2f}_batch{batch}.npz")

    np.savez_compressed(save_path,
                        old_positions=old_pos,
                        new_positions=new_pos,
                        bounces=bounce_numbers)

    print(f" Saved: {save_path}\n")

print(f"All {batches} batches completed successfully!")
print(f"Total simulated photons: {batches * photons:,}")

