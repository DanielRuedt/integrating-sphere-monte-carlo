#define _USE_MATH_DEFINES
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <cstring>

namespace py = pybind11;
using array3d = std::array<double, 3>;

// Thread-local RNG
thread_local std::mt19937 rng{std::random_device{}()};

// ===================================================================
// Rodrigues' rotation formula
// ===================================================================
array3d rotate_vector(const array3d& v, const array3d& k, double angle_deg) {
    double theta = angle_deg * M_PI / 180.0;
    double c = std::cos(theta);
    double s = std::sin(theta);
    double omc = 1.0 - c;

    return {
        v[0]*(c + k[0]*k[0]*omc) + v[1]*(k[0]*k[1]*omc - k[2]*s) + v[2]*(k[0]*k[2]*omc + k[1]*s),
        v[0]*(k[1]*k[0]*omc + k[2]*s) + v[1]*(c + k[1]*k[1]*omc) + v[2]*(k[1]*k[2]*omc - k[0]*s),
        v[0]*(k[2]*k[0]*omc - k[1]*s) + v[1]*(k[2]*k[1]*omc + k[0]*s) + v[2]*(c + k[2]*k[2]*omc)
    };
}

// ===================================================================
// Compute new position on sphere (standard untilted case)
// ===================================================================
array3d compute_new_position(const array3d& position, double alpha, 
                             double phi_direction, double R_sphere) {
    
    array3d r_hat = {position[0]/R_sphere, position[1]/R_sphere, position[2]/R_sphere};
    array3d inward = {-r_hat[0], -r_hat[1], -r_hat[2]};

    array3d arbitrary = (std::abs(inward[2]) < 0.9) 
                        ? array3d{0.0, 0.0, 1.0} : array3d{1.0, 0.0, 0.0};

    array3d e1 = {
        inward[1]*arbitrary[2] - inward[2]*arbitrary[1],
        inward[2]*arbitrary[0] - inward[0]*arbitrary[2],
        inward[0]*arbitrary[1] - inward[1]*arbitrary[0]
    };
    double norm_e1 = std::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
    if (norm_e1 > 1e-12) {
        e1[0] /= norm_e1; e1[1] /= norm_e1; e1[2] /= norm_e1;
    }

    array3d e2 = {
        inward[1]*e1[2] - inward[2]*e1[1],
        inward[2]*e1[0] - inward[0]*e1[2],
        inward[0]*e1[1] - inward[1]*e1[0]
    };

    double sa = std::sin(alpha);
    double ca = std::cos(alpha);
    double sp = std::sin(phi_direction);
    double cp = std::cos(phi_direction);

    array3d dir_local = {sa * cp, sa * sp, ca};

    array3d direction = {
        dir_local[0]*e1[0] + dir_local[1]*e2[0] + dir_local[2]*inward[0],
        dir_local[0]*e1[1] + dir_local[1]*e2[1] + dir_local[2]*inward[1],
        dir_local[0]*e1[2] + dir_local[1]*e2[2] + dir_local[2]*inward[2]
    };

    double t = -2.0 * (position[0]*direction[0] + position[1]*direction[1] + position[2]*direction[2]);

    array3d new_pos = {
        position[0] + t * direction[0],
        position[1] + t * direction[1],
        position[2] + t * direction[2]
    };

    double norm = std::sqrt(new_pos[0]*new_pos[0] + new_pos[1]*new_pos[1] + new_pos[2]*new_pos[2]);
    if (norm > 1e-12) {
        new_pos[0] *= R_sphere / norm;
        new_pos[1] *= R_sphere / norm;
        new_pos[2] *= R_sphere / norm;
    }
    return new_pos;
}

// ===================================================================
// Parse injection_type, with base_offsetX suffix
// ===================================================================
struct InjectionParams {
    std::string base_type;
    double tilt_angle_deg = 0.0;
};

InjectionParams parse_injection_type(const std::string& inj_type) {
    InjectionParams p;
    p.base_type = inj_type;

    size_t pos = inj_type.rfind("_offset");
    if (pos != std::string::npos && pos + 7 < inj_type.size()) {
        std::string suffix = inj_type.substr(pos + 7);
        try {
            p.tilt_angle_deg = std::stod(suffix);
            p.base_type = inj_type.substr(0, pos);
        } catch (...) {
            // Not a valid number -> keep original string as base
        }
    }
    return p;
}

// ===================================================================
// Main simulation
// ===================================================================
py::tuple simulate_photons_cpp(
    int n_photons,
    double R_sphere,
    int max_bounces,
    double reflectance,
    py::array_t<double> injection_point_py,
    const std::string& injection_type,
    py::array_t<double> port_center_py,
    bool explicit_direction = false,
    py::object explicit_injection_direction_py = py::none(),
    double explicit_setback = 0.0,
    double explicit_disc = 0.0,
    double explicit_tunnel = 0.0)
{
    auto inj = injection_point_py.unchecked<1>();
    auto port = port_center_py.unchecked<1>();

    array3d injection_point = {inj(0), inj(1), inj(2)};
    array3d port_center     = {port(0), port(1), port(2)};

    // ============================================================
    // Explicit direction handling
    // ============================================================
    array3d explicit_dir = {0.0, 0.0, 0.0};
    bool use_explicit_dir = explicit_direction;

    if (use_explicit_dir) {
        if (explicit_injection_direction_py.is_none()) {
            throw std::runtime_error(
                "explicit_direction is True but explicit_injection_direction is None"
            );
        }

        // Convert py::object to NumPy array
        py::array_t<double> dir_arr = 
            explicit_injection_direction_py.cast<py::array_t<double>>();

        auto dir = dir_arr.unchecked<1>();

        if (dir.ndim() != 1 || dir.shape(0) != 3) {
            throw std::runtime_error(
                "explicit_injection_direction must be a 1-D array of length 3"
            );
        }

        explicit_dir[0] = dir(0);
        explicit_dir[1] = dir(1);
        explicit_dir[2] = dir(2);

        // Normalize (safety)
        double norm = std::sqrt(
            explicit_dir[0]*explicit_dir[0] +
            explicit_dir[1]*explicit_dir[1] +
            explicit_dir[2]*explicit_dir[2]
        );

        if (norm < 1e-12) {
            throw std::runtime_error("explicit_injection_direction has zero length");
        }

        explicit_dir[0] /= norm;
        explicit_dir[1] /= norm;
        explicit_dir[2] /= norm;
    }

    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    std::vector<std::array<float, 3>> old_pos_vec;
    std::vector<std::array<float, 3>> new_pos_vec;
    std::vector<int> bounce_vec;

    old_pos_vec.reserve(n_photons / 10);
    new_pos_vec.reserve(n_photons / 10);
    bounce_vec.reserve(n_photons / 10);

    auto params = parse_injection_type(injection_type);
    const std::string& base = params.base_type;
    double tilt_angle = params.tilt_angle_deg;

    for (int i = 0; i < n_photons; ++i) {
        double alpha = 0.0, phi = 0.0;
        array3d pos;
        array3d old_pos = injection_point;
        int bounce = 0;

        // ====================== DIRECTION SAMPLING ======================
        if (base == "fiber" || base == "fiber_largeNA") {
            double na = (base == "fiber") ? 0.22 : 0.39;
            alpha = std::asin(na * std::sqrt(uniform(rng)));
            phi = uniform(rng) * 2.0 * M_PI;
        }
        else if (base == "diffuse" || base == "lambertian") {
            alpha = std::asin(std::sqrt(uniform(rng)));
            phi = uniform(rng) * 2.0 * M_PI;
        }
        else if (base == "laser") {
            alpha = 0.0;   // collimated
            phi = 0.0;
        }
        else {
            // fallback
            alpha = std::asin(std::sqrt(uniform(rng)));
            phi = uniform(rng) * 2.0 * M_PI;
        }

        // ====================== INITIAL POSITION ======================
        if (use_explicit_dir) {
            // ----------------------------------------------------------
            // Explicit direction: cone centered on explicit_dir
            // ----------------------------------------------------------
            array3d cone_axis = explicit_dir;   // already normalized
            
            // Build local frame around the explicit direction
            array3d arbitrary = (std::abs(cone_axis[2]) < 0.9) 
                                ? array3d{0.0, 0.0, 1.0} : array3d{1.0, 0.0, 0.0};

            array3d e1 = {
                cone_axis[1]*arbitrary[2] - cone_axis[2]*arbitrary[1],
                cone_axis[2]*arbitrary[0] - cone_axis[0]*arbitrary[2],
                cone_axis[0]*arbitrary[1] - cone_axis[1]*arbitrary[0]
            };
            double norm_e1 = std::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
            if (norm_e1 > 1e-12) {
                e1[0] /= norm_e1; e1[1] /= norm_e1; e1[2] /= norm_e1;
            }

            array3d e2 = {
                cone_axis[1]*e1[2] - cone_axis[2]*e1[1],
                cone_axis[2]*e1[0] - cone_axis[0]*e1[2],
                cone_axis[0]*e1[1] - cone_axis[1]*e1[0]
            };

            // Sample direction inside the cone (alpha/phi already computed above)
            double sa = std::sin(alpha);
            double ca = std::cos(alpha);
            double sp = std::sin(phi);
            double cp = std::cos(phi);

            array3d dir_local = {sa * cp, sa * sp, ca};

            array3d direction = {
                dir_local[0]*e1[0] + dir_local[1]*e2[0] + dir_local[2]*cone_axis[0],
                dir_local[0]*e1[1] + dir_local[1]*e2[1] + dir_local[2]*cone_axis[1],
                dir_local[0]*e1[2] + dir_local[1]*e2[2] + dir_local[2]*cone_axis[2]
            };

            // ================================================================
            // First-hit calculation (with optional setback + disc)
            // ================================================================
            if (explicit_setback > 0.0) {
                // Nominal emission origin outside of the sphere 
                array3d setback_point = {
                    injection_point[0] - explicit_setback * explicit_dir[0],
                    injection_point[1] - explicit_setback * explicit_dir[1],
                    injection_point[2] - explicit_setback * explicit_dir[2]
                };

                // Optional randomisation inside a disc of radius explicit_disc around the emission origin 
                if (explicit_disc > 0.0) {
                    // Build orthonormal frame perpendicular to explicit_dir
                    array3d arbitrary = (std::abs(explicit_dir[2]) < 0.9)
                                        ? array3d{0.0, 0.0, 1.0}
                                        : array3d{1.0, 0.0, 0.0};

                    array3d e1 = {
                        explicit_dir[1]*arbitrary[2] - explicit_dir[2]*arbitrary[1],
                        explicit_dir[2]*arbitrary[0] - explicit_dir[0]*arbitrary[2],
                        explicit_dir[0]*arbitrary[1] - explicit_dir[1]*arbitrary[0]
                    };
                    double norm_e1 = std::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
                    if (norm_e1 > 1e-12) {
                        e1[0] /= norm_e1; e1[1] /= norm_e1; e1[2] /= norm_e1;
                    }

                    array3d e2 = {
                        explicit_dir[1]*e1[2] - explicit_dir[2]*e1[1],
                        explicit_dir[2]*e1[0] - explicit_dir[0]*e1[2],
                        explicit_dir[0]*e1[1] - explicit_dir[1]*e1[0]
                    };

                    // Uniform point inside the disc
                    double r     = explicit_disc * std::sqrt(uniform(rng));
                    double theta = uniform(rng) * 2.0 * M_PI;
                    double ct = std::cos(theta);
                    double st = std::sin(theta);

                    setback_point[0] += r * (ct * e1[0] + st * e2[0]);
                    setback_point[1] += r * (ct * e1[1] + st * e2[1]);
                    setback_point[2] += r * (ct * e1[2] + st * e2[2]);

                    // Safety: the new origin of emission must still lie outside the sphere
                    double r2 = setback_point[0]*setback_point[0] +
                                setback_point[1]*setback_point[1] +
                                setback_point[2]*setback_point[2];
                    if (r2 < R_sphere * R_sphere) {
                        throw std::runtime_error(
                            "explicit_setback is not large enough for the chosen explicit_disc radius"
                        );
                    }
                }

                // Optional diffuse injection, i.e. diffuse reflection of injected photons, from the tunnel walls 
                if (explicit_tunnel > 0.0) {
                    // Build orthonormal frame perpendicular to explicit_dir
                    array3d arbitrary = (std::abs(explicit_dir[2]) < 0.9)
                                        ? array3d{0.0, 0.0, 1.0}
                                        : array3d{1.0, 0.0, 0.0};

                    array3d e1 = {
                        explicit_dir[1]*arbitrary[2] - explicit_dir[2]*arbitrary[1],
                        explicit_dir[2]*arbitrary[0] - explicit_dir[0]*arbitrary[2],
                        explicit_dir[0]*arbitrary[1] - explicit_dir[1]*arbitrary[0]
                    };
                    double norm_e1 = std::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
                    if (norm_e1 > 1e-12) {
                        e1[0] /= norm_e1; e1[1] /= norm_e1; e1[2] /= norm_e1;
                    }

                    array3d e2 = {
                        explicit_dir[1]*e1[2] - explicit_dir[2]*e1[1],
                        explicit_dir[2]*e1[0] - explicit_dir[0]*e1[2],
                        explicit_dir[0]*e1[1] - explicit_dir[1]*e1[0]
                    };

                    // We randomly sample a point on the tunnel wall by sampling from the surface of a cylinder with radius explicit_tunnel and cylinder length 3 * explicit_setback 
                    double r     = explicit_tunnel;
                    double theta = uniform(rng) * 2.0 * M_PI;
                    double ct = std::cos(theta);
                    double st = std::sin(theta);

                    array3d original_setback_point = setback_point;  // Store original for safety check
                    double random_delta = 3 * explicit_setback * uniform(rng); 

                    setback_point[0] = original_setback_point[0] + r * (ct * e1[0] + st * e2[0]) + random_delta * explicit_dir[0];
                    setback_point[1] = original_setback_point[1] + r * (ct * e1[1] + st * e2[1]) + random_delta * explicit_dir[1];
                    setback_point[2] = original_setback_point[2] + r * (ct * e1[2] + st * e2[2]) + random_delta * explicit_dir[2];

                    // Safety: the new origin of emission must still lie outside the sphere
                    double r2 = setback_point[0]*setback_point[0] +
                                setback_point[1]*setback_point[1] +
                                setback_point[2]*setback_point[2];

                    while (r2 < R_sphere * R_sphere) {
                        // Adjust the setback point until it lies outside the sphere
                        r = explicit_tunnel;
                        theta = uniform(rng) * 2.0 * M_PI;
                        ct = std::cos(theta);
                        st = std::sin(theta);
                        random_delta = 3 * explicit_setback * uniform(rng);
                        setback_point[0] = original_setback_point[0] + r * (ct * e1[0] + st * e2[0]) + random_delta * explicit_dir[0];
                        setback_point[1] = original_setback_point[1] + r * (ct * e1[1] + st * e2[1]) + random_delta * explicit_dir[1];
                        setback_point[2] = original_setback_point[2] + r * (ct * e1[2] + st * e2[2]) + random_delta * explicit_dir[2];
                        r2 = setback_point[0]*setback_point[0] +
                             setback_point[1]*setback_point[1] +
                             setback_point[2]*setback_point[2];
                    }
                    
                    // ---------------------------------------------------------------
                    // Re-assign direction to Lambertian reflection from the tunnel wall
                    // ---------------------------------------------------------------
                    
                    // Vector from the cylinder axis to the emission point
                    // (original_setback_point lies on the axis)
                    array3d to_point = {
                        setback_point[0] - original_setback_point[0],
                        setback_point[1] - original_setback_point[1],
                        setback_point[2] - original_setback_point[2]
                    };

                    // Remove the axial component to get the pure radial vector
                    double axial = to_point[0]*explicit_dir[0] + to_point[1]*explicit_dir[1] + to_point[2]*explicit_dir[2];
                    array3d radial = {
                        to_point[0] - axial * explicit_dir[0],
                        to_point[1] - axial * explicit_dir[1],
                        to_point[2] - axial * explicit_dir[2]
                    };
                    
                    array3d tang1 = {0.0, 0.0, 0.0};
                    array3d tang2 = {0.0, 0.0, 0.0};
                    array3d normal = {0.0, 0.0, 0.0};

                    double rlen = std::sqrt(radial[0]*radial[0] + radial[1]*radial[1] + radial[2]*radial[2]);
                    if (rlen < 1e-12) {
                        // Degenerate (should never happen if explicit_tunnel is chosen sensibly),  go to next photon
                        continue;
                    } else {
                        // Inward-pointing normal (toward the axis)
                        normal = {
                            -radial[0] / rlen,
                            -radial[1] / rlen,
                            -radial[2] / rlen
                        };

                        // Local orthonormal frame around the normal
                        array3d arbitrary = (std::abs(normal[2]) < 0.9)
                                            ? array3d{0.0, 0.0, 1.0}
                                            : array3d{1.0, 0.0, 0.0};

                        tang1 = {
                            normal[1]*arbitrary[2] - normal[2]*arbitrary[1],
                            normal[2]*arbitrary[0] - normal[0]*arbitrary[2],
                            normal[0]*arbitrary[1] - normal[1]*arbitrary[0]
                        };
                        double nt1 = std::sqrt(tang1[0]*tang1[0] + tang1[1]*tang1[1] + tang1[2]*tang1[2]);
                        if (nt1 > 1e-12) {
                            tang1[0] /= nt1; tang1[1] /= nt1; tang1[2] /= nt1;
                        }

                        tang2 = {
                            normal[1]*tang1[2] - normal[2]*tang1[1],
                            normal[2]*tang1[0] - normal[0]*tang1[2],
                            normal[0]*tang1[1] - normal[1]*tang1[0]
                        };
                    }

                    bool valid_direction = false;
                    const double tol = 1e-6;   // numerical tolerance

                    while (!valid_direction) {

                        // ----- sample Lambertian direction (unchanged) -----
                        double alpha_lam = std::asin(std::sqrt(uniform(rng)));
                        double phi_lam   = uniform(rng) * 2.0 * M_PI;
                        double sa = std::sin(alpha_lam);
                        double ca = std::cos(alpha_lam);
                        double sp = std::sin(phi_lam);
                        double cp = std::cos(phi_lam);
                        array3d dir_local = {sa * cp, sa * sp, ca};

                        direction = {
                            dir_local[0]*tang1[0] + dir_local[1]*tang2[0] + dir_local[2]*normal[0],
                            dir_local[0]*tang1[1] + dir_local[1]*tang2[1] + dir_local[2]*normal[1],
                            dir_local[0]*tang1[2] + dir_local[1]*tang2[2] + dir_local[2]*normal[2]
                        };

                        // ----- ray-sphere quadratic (near intersection) -----
                        double B = 2.0 * (setback_point[0]*direction[0] +
                                          setback_point[1]*direction[1] +
                                          setback_point[2]*direction[2]);
                        double C = (setback_point[0]*setback_point[0] +
                                    setback_point[1]*setback_point[1] +
                                    setback_point[2]*setback_point[2]) - R_sphere * R_sphere;
                        double disc = B*B - 4.0 * C;

                        if (disc < 0.0) continue;               // misses sphere entirely

                        double sqrt_disc = std::sqrt(disc);
                        double t_near = (-B - sqrt_disc) * 0.5; // the smaller root
                        if (t_near <= 0.0) continue;            // intersection behind the ray

                        // Intersection point
                        array3d hit = {
                            setback_point[0] + t_near * direction[0],
                            setback_point[1] + t_near * direction[1],
                            setback_point[2] + t_near * direction[2]
                        };

                        // Perpendicular distance from hit to the tunnel axis
                        // Axis point = original_setback_point, direction = explicit_dir
                        array3d to_hit = {
                            hit[0] - original_setback_point[0],
                            hit[1] - original_setback_point[1],
                            hit[2] - original_setback_point[2]
                        };
                        double axial = to_hit[0]*explicit_dir[0] + to_hit[1]*explicit_dir[1] + to_hit[2]*explicit_dir[2];
                        array3d radial_vec = {
                            to_hit[0] - axial * explicit_dir[0],
                            to_hit[1] - axial * explicit_dir[1],
                            to_hit[2] - axial * explicit_dir[2]
                        };
                        double dist_to_axis = std::sqrt(radial_vec[0]*radial_vec[0] +
                                                        radial_vec[1]*radial_vec[1] +
                                                        radial_vec[2]*radial_vec[2]);

                        if (dist_to_axis <= explicit_tunnel + tol) {
                            valid_direction = true;   // ray stays inside the tunnel until it enters the sphere
                        }
                    }
                }

                // Full ray-sphere quadratic to far intersection
                double B = 2.0 * (setback_point[0]*direction[0] +
                                  setback_point[1]*direction[1] +
                                  setback_point[2]*direction[2]);
                double C = (setback_point[0]*setback_point[0] +
                            setback_point[1]*setback_point[1] +
                            setback_point[2]*setback_point[2]) - R_sphere * R_sphere;

                double disc = B*B - 4.0 * C;

                if (disc < 0.0) {
                    continue;   // ray misses the sphere
                }

                double sqrt_disc = std::sqrt(disc);
                double t1 = (-B - sqrt_disc) * 0.5;
                double t2 = (-B + sqrt_disc) * 0.5;

                double t_far = std::max(t1, t2);
                if (t_far <= 0.0) {
                    continue;   // the far intersection has to lie in the injection direction 
                }

                pos = {
                    setback_point[0] + t_far * direction[0],
                    setback_point[1] + t_far * direction[1],
                    setback_point[2] + t_far * direction[2]
                };

                // Make sure to use the actual injection origin as old position when photons exit the sphere at bounce 0 
                old_pos = setback_point;
            }
            else {
                // Original on-surface formula (no setback)
                double t = -2.0 * (injection_point[0]*direction[0] +
                                   injection_point[1]*direction[1] +
                                   injection_point[2]*direction[2]);

                pos = {
                    injection_point[0] + t * direction[0],
                    injection_point[1] + t * direction[1],
                    injection_point[2] + t * direction[2]
                };
            }

            // Re-normalize for numerical safety (always)
            double norm = std::sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]);
            if (norm > 1e-12) {
                pos[0] *= R_sphere / norm;
                pos[1] *= R_sphere / norm;
                pos[2] *= R_sphere / norm;
            }
        }
        else if (tilt_angle == 0.0) {
            // Standard case
            pos = compute_new_position(injection_point, alpha, phi, R_sphere);
        }
        else {
            // ====================== TILTED CONE ======================
            array3d r_hat = {injection_point[0]/R_sphere,
                             injection_point[1]/R_sphere,
                             injection_point[2]/R_sphere};
            array3d nominal_inward = {-r_hat[0], -r_hat[1], -r_hat[2]};

            array3d global_down = {0.0, 0.0, -1.0};
            array3d tilt_axis = {
                nominal_inward[1]*global_down[2] - nominal_inward[2]*global_down[1],
                nominal_inward[2]*global_down[0] - nominal_inward[0]*global_down[2],
                nominal_inward[0]*global_down[1] - nominal_inward[1]*global_down[0]
            };
            double n_axis = std::sqrt(tilt_axis[0]*tilt_axis[0] + 
                                     tilt_axis[1]*tilt_axis[1] + 
                                     tilt_axis[2]*tilt_axis[2]);
            if (n_axis > 1e-12) {
                tilt_axis[0] /= n_axis; tilt_axis[1] /= n_axis; tilt_axis[2] /= n_axis;
            } else {
                tilt_axis = {1.0, 0.0, 0.0};
            }

            array3d tilted_inward = rotate_vector(nominal_inward, tilt_axis, tilt_angle);
            double n_tilt = std::sqrt(tilted_inward[0]*tilted_inward[0] + 
                                     tilted_inward[1]*tilted_inward[1] + 
                                     tilted_inward[2]*tilted_inward[2]);
            if (n_tilt > 1e-12) {
                tilted_inward[0] /= n_tilt;
                tilted_inward[1] /= n_tilt;
                tilted_inward[2] /= n_tilt;
            }

            // Local frame around tilted axis
            array3d arbitrary = (std::abs(tilted_inward[2]) < 0.9) 
                                ? array3d{0.0, 0.0, 1.0} : array3d{1.0, 0.0, 0.0};

            array3d e1 = {
                tilted_inward[1]*arbitrary[2] - tilted_inward[2]*arbitrary[1],
                tilted_inward[2]*arbitrary[0] - tilted_inward[0]*arbitrary[2],
                tilted_inward[0]*arbitrary[1] - tilted_inward[1]*arbitrary[0]
            };
            double norm_e1 = std::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
            if (norm_e1 > 1e-12) {
                e1[0] /= norm_e1; e1[1] /= norm_e1; e1[2] /= norm_e1;
            }

            array3d e2 = {
                tilted_inward[1]*e1[2] - tilted_inward[2]*e1[1],
                tilted_inward[2]*e1[0] - tilted_inward[0]*e1[2],
                tilted_inward[0]*e1[1] - tilted_inward[1]*e1[0]
            };

            double sa = std::sin(alpha);
            double ca = std::cos(alpha);
            double sp = std::sin(phi);
            double cp = std::cos(phi);

            array3d dir_local = {sa * cp, sa * sp, ca};

            array3d direction = {
                dir_local[0]*e1[0] + dir_local[1]*e2[0] + dir_local[2]*tilted_inward[0],
                dir_local[0]*e1[1] + dir_local[1]*e2[1] + dir_local[2]*tilted_inward[1],
                dir_local[0]*e1[2] + dir_local[1]*e2[2] + dir_local[2]*tilted_inward[2]
            };

            double t = -2.0 * (injection_point[0]*direction[0] + 
                               injection_point[1]*direction[1] + 
                               injection_point[2]*direction[2]);

            pos = {
                injection_point[0] + t * direction[0],
                injection_point[1] + t * direction[1],
                injection_point[2] + t * direction[2]
            };

            double norm = std::sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]);
            if (norm > 1e-12) {
                pos[0] *= R_sphere / norm;
                pos[1] *= R_sphere / norm;
                pos[2] *= R_sphere / norm;
            }
        }

        // ====================== PORT CHECK ======================
        if (pos[2] <= port_center[2]) {
            old_pos_vec.push_back({(float)old_pos[0], (float)old_pos[1], (float)old_pos[2]});
            new_pos_vec.push_back({(float)pos[0], (float)pos[1], (float)pos[2]});
            bounce_vec.push_back(bounce);
            continue;
        }

        if (uniform(rng) > reflectance) continue;

        // ====================== BOUNCE LOOP ======================
        for (bounce = 1; bounce <= max_bounces; ++bounce) {
            alpha = std::asin(std::sqrt(uniform(rng)));
            phi = uniform(rng) * 2.0 * M_PI;

            old_pos = pos;
            pos = compute_new_position(old_pos, alpha, phi, R_sphere);

            if (pos[2] <= port_center[2]) {
                old_pos_vec.push_back({(float)old_pos[0], (float)old_pos[1], (float)old_pos[2]});
                new_pos_vec.push_back({(float)pos[0], (float)pos[1], (float)pos[2]});
                bounce_vec.push_back(bounce);
                break;
            }

            if (uniform(rng) > reflectance) break;
        }
    }

    // ====================== CONVERT TO NUMPY ======================
    py::ssize_t n = static_cast<py::ssize_t>(old_pos_vec.size());

    std::vector<py::ssize_t> shape2d = {n, 3};
    std::vector<py::ssize_t> shape1d = {n};

    py::array_t<float> np_old(shape2d);
    py::array_t<float> np_new(shape2d);
    py::array_t<int>   np_bounces(shape1d);

    if (n > 0) {
        std::memcpy(np_old.mutable_data(), old_pos_vec.data(), n * 3 * sizeof(float));
        std::memcpy(np_new.mutable_data(), new_pos_vec.data(), n * 3 * sizeof(float));
        std::memcpy(np_bounces.mutable_data(), bounce_vec.data(), n * sizeof(int));
    }

    return py::make_tuple(np_old, np_new, np_bounces);
}

// ===================================================================
PYBIND11_MODULE(sphere_sim_cpp, m) {
    m.def("simulate_photons", &simulate_photons_cpp,
          py::arg("n_photons"), py::arg("R_sphere"), py::arg("max_bounces"),
          py::arg("reflectance"), py::arg("injection_point"),
          py::arg("injection_type"), py::arg("port_center"),
          py::arg("explicit_direction") = false,
          py::arg("explicit_injection_direction") = py::none(),
          py::arg("explicit_setback") = 0.0,
          py::arg("explicit_disc") = 0.0,
          py::arg("explicit_tunnel") = 0.0);
}