# Integrating Sphere Monte Carlo Simulation

Monte Carlo simulation of light propagation in an integrating sphere, together with analysis scripts.

## Files

- **`cpp_sphere_sim.py`** – Main Monte Carlo simulation script  
  (contains detailed comments explaining the algorithm, Lambertian reflection, sphere geometry, etc.)

- **`sphere_sim.cpp`** – C++ module called inside the Python script to run the actual Monte Carlo simulation loop.

- **`sphere_analysis_multi_batch.py`** – Post-processing of the simulation results. Combines all batch files of one
  simulation and produces figures of the bounce statistics, exit-angle distributions, photon distribution in the port
  plane, port uniformity, and the deviation of the exit-angle distribution in the central 0.5 mm × 0.5 mm square of
  the port from an ideal Lambertian distribution as a function of NA. The deviation data is saved as
  `*_all_batches_deviation_vs_NA.npz` for use by the two scripts below.

- **`combined_deviation_NA_any_variable.py`** – Compares the deviation from Lambertian vs. NA for several simulations
  that differ in one parameter (e.g. reflectance, injection NA or injection angle offset), including the deviation at
  fixed objective NAs and the relative photon count in the central square.

- **`combined_deviation_NA_mixtures.py`** – Combines the deviation curves of two simulations (directional fiber
  injection and diffuse injection from the tunnel wall) as weighted mixtures. The weights are the fractions of
  detected photons in the central square coming from each injection type (e.g. 5 % diffuse = 5 % of the detected
  photons originate from diffuse injection).

## Requirements

- Python 3 with `numpy`, `matplotlib`, `pybind11` and `setuptools` (`pip install numpy matplotlib pybind11 setuptools`)
- A C++17 compiler:
  - macOS: Apple Clang (Xcode Command Line Tools: `xcode-select --install`)
  - Linux: GCC or Clang
  - Windows: MSVC, included in the free Visual Studio Build Tools (select the workload
    "Desktop development with C++" during installation). MinGW-w64 toolchains are not
    recommended; if you use one, it must be based on GCC 9.2 or newer.

## Building the C++ module

`sphere_sim.cpp` is a pybind11 extension and must be compiled before running the simulation.
The compiled module **must be named `sphere_sim_cpp`**, as defined by `PYBIND11_MODULE(sphere_sim_cpp, m)`
in the source and imported by `cpp_sphere_sim.py`, with the file extension Python expects on your platform
(e.g. `sphere_sim_cpp.cpython-312-darwin.so` on macOS, `sphere_sim_cpp.cp312-win_amd64.pyd` on Windows).

Place the compiled module in the same folder as `cpp_sphere_sim.py`. It works only with the Python
version it was compiled for, so recompile after switching Python versions.

### Windows

Open a Command Prompt or PowerShell in the repository folder and run (one line):

```
python -c "from setuptools import setup; from pybind11.setup_helpers import Pybind11Extension; setup(name='sphere_sim_cpp', ext_modules=[Pybind11Extension('sphere_sim_cpp', ['sphere_sim.cpp'], cxx_std=17)], script_args=['build_ext', '--inplace'])"
```

This finds the MSVC compiler automatically and places `sphere_sim_cpp.cp3XX-win_amd64.pyd`
in the current folder. The same command also works on macOS and Linux (use `python3` there).

### macOS

```bash
c++ -O3 -Wall -shared -std=c++17 -undefined dynamic_lookup \
    $(python3 -m pybind11 --includes) sphere_sim.cpp \
    -o sphere_sim_cpp$(python3-config --extension-suffix)
```

### Linux

```bash
c++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) sphere_sim.cpp \
    -o sphere_sim_cpp$(python3-config --extension-suffix)
```

## Usage

1. Compile `sphere_sim.cpp` as described above.
2. Set the simulation parameters at the top of `cpp_sphere_sim.py` and run it. The results are saved as
   `*_batch1.npz`, `*_batch2.npz`, ... in the script folder.
3. In `sphere_analysis_multi_batch.py`, set `base_name` to the common file name of the batch files, ending in
   `_batch` (e.g. `photon_exit_data_10MPhotons_reflectance98_IC2_fiber_explicit_setback4_batch`), and run it.
   All batches are detected automatically. The figures are saved as PDFs, together with the
   `*_all_batches_deviation_vs_NA.npz` file. The output file names assume 10 million photons per batch.
4. Optional: list the `*_deviation_vs_NA.npz` files of several simulations in
   `combined_deviation_NA_any_variable.py` or `combined_deviation_NA_mixtures.py` and run them to compare the results.

Run all scripts from the repository folder, so that the output files of each step are found by the next one.

Author: Daniel Ruedt, B.Sc. Electrical Engineering and Information Technology
