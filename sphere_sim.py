from unittest import case

import scipy as sp
import matplotlib.pyplot as plt
import numpy as np 
import numba as nb
import os
from joblib import Parallel, delayed
import gc


# This code simulates the photon flux generated at the port of an integrating sphere. It assumes perfect lambertian reflection on the inner sphere 
# surface. The reflectance is modelled explicitly by the reflectance value and the probability of absorption at every bounce. The probability of absorption is 1 - P(reflection). 
# For the reflection, the polar angle relative to the surface normal and the azimuthal angle are calculated using the method described in 
# Prokhorov et al. 2003, "Monte Carlo modeling of an integrating sphere reflectometer" DOI: 10.1364/AO.42.003832 , https://www.virial.com/pdf/AO2003-42-19-3832-3842_IS.pdf 

@nb.njit(fastmath=True)
def compute_new_position(position, alpha, phi_direction, R_sphere):
    """
    Compute next intersection point on sphere after Lambertian-like reflection.
    
    Parameters:
    - position: np.array([x, y, z]) current point on sphere (Radius = R_sphere)
    - alpha: float, angle from inward normal (0 = toward center, pi/2 = tangent)
    - phi_direction: float, azimuthal angle in local tangent plane (0 to 2*pi)
    
    Returns:
    - np.array([x, y, z]) next point on sphere surface
    """
    r_hat = position / R_sphere                            # outward unit normal
    
    # Inward normal (our reference direction for alpha=0)
    inward = -r_hat
    
    # Build local orthonormal basis at the point
    # We need two vectors perpendicular to inward
    
    # Arbitrary vector not parallel to inward
    arbitrary = np.array([0, 0, 1]) if abs(inward[2]) < 0.9 else np.array([1, 0, 0])
    
    # First tangent vector
    e1 = np.cross(inward, arbitrary)
    e1 /= np.linalg.norm(e1)
    
    # Second tangent vector (right-handed)
    e2 = np.cross(inward, e1)
    
    # Direction vector in local frame
    dir_local = np.array([
        np.sin(alpha) * np.cos(phi_direction),   # e1 component
        np.sin(alpha) * np.sin(phi_direction),   # e2 component
        np.cos(alpha)                            # inward component
    ])
    
    # Global direction
    direction = dir_local[0]*e1 + dir_local[1]*e2 + dir_local[2]*inward
    
    # Find second intersection with sphere
    # absolute value of the position vector = R_sphere, absolute value of the direction vector = 1. The minus sign ensures t is non-negative. 
    t = -2.0 * np.dot(position, direction)
    
    new_position = position + t * direction
    new_position *= R_sphere / np.linalg.norm(new_position)   # Project back exactly on sphere
    
    return new_position

def simulate_one_photon(photon_id, R_sphere, bounces, reflectance, 
                       injection_point, injection_type, port_center):
    
    # Initial injection
    injection_type_val = np.array([0.0, 0.0])  # This array is used to store the value of alpha (= polar angle) and phi_direction (= azimuthal angle) for the initial injection

    if injection_type == "laser":
        injection_type_val = np.array([0.0, 0.0]) # alpha = 0 (straight through the center of the sphere), phi_direction = 0 (arbitrary since alpha=0)

    elif injection_type == "fiber":
        injection_type_val[0] = np.arcsin(0.20*np.sqrt(np.random.uniform(0, 1)))     # NA = 0.20 corresponds to max alpha of 11.5 degrees, this implements a truncated lambertian distribution 
        injection_type_val[1] = np.random.uniform(0, 2*np.pi)

    elif injection_type == "fiber_largeNA":
        injection_type_val[0] = np.arcsin(0.35*np.sqrt(np.random.uniform(0, 1)))     # NA = 0.35 corresponds to max alpha of 20.1 degrees, this implements a truncated lambertian distribution 
        injection_type_val[1] = np.random.uniform(0, 2*np.pi)

    else:                                    # default is diffuse injection 
        injection_type_val[0] = np.arcsin(np.sqrt(np.random.uniform(0, 1)))
        injection_type_val[1] = np.random.uniform(0, 2*np.pi)

    old_position = injection_point.copy() 
    photon_position = compute_new_position(injection_point, injection_type_val[0], injection_type_val[1], R_sphere)
    
    bounce = 0  # Initial injection is considered bounce 0 

    # Check if the photon has actually left the sphere by going through the port and now having a z coordinate that is less than the z coordinate of the port center. 
    if photon_position[2] <= port_center[2]:
        # print("Photon " + str(photon_id) + " exited at injection (bounce 0)")
        return (np.int32(photon_id), old_position.astype(np.float32), photon_position.astype(np.float32), np.int32(bounce))     # Go to next photon 

    if np.random.uniform(0, 1) > reflectance: 
        # print("Photon " + str(photon_id) + " absorbed directly after injection (bounce 0) with coordinates: " + str(photon_position)) 
        return None                     # Go to next photon

    for bounce in range(1, bounces + 1):
        
        alpha = np.arcsin(np.sqrt(np.random.uniform(0, 1)))
        phi_direction = np.random.uniform(0, 2*np.pi)
        
        old_position = photon_position.copy()
        photon_position = compute_new_position(photon_position, alpha, phi_direction, R_sphere)
        
        if photon_position[2] <= port_center[2]:
            return (np.int32(photon_id), old_position.astype(np.float32), photon_position.astype(np.float32), np.int32(bounce))
        
        if np.random.uniform(0, 1) > reflectance:
            # print("Photon " + str(photon_id) + " absorbed after bounce " + str(bounce) + " with coordinates: " + str(photon_position))
            return None  # absorbed
        
    return None  # reached max bounces


# ====================== BEGINNING SIMULATION PARAMETERS ======================
bounces = 460 # Maximum bounces before terminating a photon.
              # Chosen so that reflectance^n < 0.01 (i.e. >99% absorbed on average)
              # Values: 460 for reflectance=0.99, 90 for 0.95, 44 for 0.90, 
              # 29 for 0.85, 21 for 0.80
reflectance = 0.99 # Reflectance of the sphere 
batches = 10 # Number of batches to split the simulation into. 
photons = 10*(10**6) # Number of photons per batch. Total photons simulated will be photons * batches. 

sphere_diameter = 1.75*25.4 # Diameter of the sphere in milli meters converted from inches, 1.75 inch is the diameter of the integrating sphere in the IC2 integrating cube from StellarNet.
R_sphere = sphere_diameter/2 # Radius of the sphere

port_diameter = (5/8)*25.4 # Diameter of the port in milli meters converted from inches, 5/8 inch is the port diameter for the IC2 integrating cube from StellarNet. 
r_port = port_diameter/2 # Radius of the port

origin = np.array([0,0,0]) # Origin of the cartesian coordinate system x, y, z. The center of the sphere is at the origin. 

# Point of injection of the photon, defined in cartesian coordinates as one of the corners of the cube surrounding the sphere. 
injection_point = np.array([R_sphere/np.sqrt(3),R_sphere/np.sqrt(3),R_sphere/np.sqrt(3)])
injection_type = "fiber" # Type of injection, either "laser", "fiber", "diffuse" or "fiber_largeNA". This determines the range of angles of injection for the photons. laser = one angle 


alpha_port = np.arcsin(r_port/R_sphere) # Angle relative to the normal of the port surface subtended by the port at the bottom of the sphere in radians 
port_center = np.array([0,0, -R_sphere * np.cos(alpha_port)]) # Center of the port. 
# ===================== END SIMULATION PARAMETERS ====================== 


print("Simulating photon flux at the port of an integrating sphere with a reflectance of " + str(reflectance*100) + "%") 
print(f"Starting parallel simulation of {batches * photons:,} photons in {batches} batches...")

for batch in range(1, batches+1): 
    print(f"\nBatch {batch}/{batches} - Simulating {photons:,} photons...")

    results = Parallel(n_jobs=6, verbose=6)(
        delayed(simulate_one_photon)(i, R_sphere, bounces, reflectance, injection_point, injection_type, port_center)
        for i in range(1, photons+1)
    )


    # Pre-allocate arrays to store the data. We store 3 variables: the old position, i.e. the last position of the photon before it exited the sphere, the new position, 
    # i.e. the position that the photon would be in if the sphere continued below the exit port, and finally the number of bounces that the photon underwent before exiting. 
    exit_old_positions = np.zeros((photons, 3), dtype=np.float32)
    exit_new_positions = np.zeros((photons, 3), dtype=np.float32)
    exit_bounce_numbers = np.zeros(photons, dtype=np.int32)
    exit_count = 0   # counter to keep track of how many actually exited


    # Fill the arrays from the results
    exit_count = 0
    for res in results:
        if res is not None:
            photon_id, old_pos, new_pos, bounce = res
            exit_old_positions[exit_count] = old_pos
            exit_new_positions[exit_count] = new_pos
            exit_bounce_numbers[exit_count] = bounce
            exit_count += 1


    del results  # Free memory as results is no longer needed
    gc.collect()

    # Trim the arrays to actual number of exited photons
    exit_old_positions = exit_old_positions[:exit_count]
    exit_new_positions = exit_new_positions[:exit_count]
    exit_bounce_numbers = exit_bounce_numbers[:exit_count]


    print(f"Total photons exited through port: {exit_count} / {photons}")
    print("Old positions for the first 10 escaped photons:")
    for i in range(min(10, exit_count)):
        print(f"  {exit_old_positions[i]}")


    # Save data to disk
    sim_photon_number_millions = photons // 1_000_000
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if sphere_diameter/port_diameter == 1.75/(5/8): # 1.75/(5/8) is the ratio for the IC2 integrating cube from StellarNet 
        save_path = os.path.join(script_dir, "photon_exit_data" + f"_{sim_photon_number_millions}MPhotons" + f"_reflectance{int(reflectance*100)}_{injection_type}_batch{batch}" + ".npz")
    else: 
        save_path = os.path.join(script_dir, "photon_exit_data" + f"_{sim_photon_number_millions}MPhotons" + f"_reflectance{int(reflectance*100)}_{injection_type}" + 
                                 f"_diameter_port_ratio_{sphere_diameter/port_diameter:.2f}_batch{batch}" + ".npz")
    
    np.savez_compressed(save_path, 
                        old_positions=exit_old_positions,
                        new_positions=exit_new_positions,
                        bounces=exit_bounce_numbers)

    print(f"Saved: {save_path}")

print(f"\nSimulated {batches * photons:,} photons in {batches} batches successfully!") 

