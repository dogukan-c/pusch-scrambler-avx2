# Toolchain setup

Intel oneAPI `icpx` is the default. Any C++17 compiler with AVX2 support works.

```bash
make                 # icpx
make CXX=g++
make CXX=clang++
```

## Intel oneAPI (icpx) on Ubuntu

`apt` no longer carries this compiler on Ubuntu 22.04, so use the offline
installer:

1. Download from
   <https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler-download.html?operatingsystem=linux&distribution-linux=offline>
2. `chmod +x ./installer-name && ./installer-name`
3. Load the environment in every shell that builds:
   ```bash
   source /opt/intel/oneapi/setvars.sh
   ```
4. `make CXX=icpx`

## MATLAB

`matlab/verify_with_matlab.m` needs the **5G Toolbox** (for `nrPUSCHScramble`).

```matlab
cd matlab
verify_with_matlab
```

## Regenerating the test vectors

```bash
make vectors        # needs python3; rewrites test/vectors/*.ref
```

Existing `.in` payloads are reused rather than re-randomised, so committed
vectors stay byte-stable.
