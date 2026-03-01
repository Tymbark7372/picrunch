# picrunch

multi-backend pi calculator that computes billions of digits using CUDA, OpenCL, and CPU

## what it does

calculates pi using the [chudnovsky algorithm](https://en.wikipedia.org/wiki/Chudnovsky_algorithm) with GMP big-number arithmetic. supports three compute backends that never mix - pick one and it does everything on that backend:

- **cuda** - custom base-10^9 GPU kernels, warp-shuffle reductions, NVML stats
- **opencl** - portable GPU compute, works on AMD/Intel/NVIDIA
- **cpu** - OpenMP parallelism with optional core pinning, binary splitting variant for large digit counts

also does numerical integration (midpoint rule + kahan summation) for quick double-precision pi if you don't need digits

## building

### linux

```bash
make
```

or with cmake:

```bash
cmake -B build
cmake --build build
```

### windows

```bash
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
```

target a specific GPU arch for faster builds:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89
```

### dependencies

| dependency | what for | package (ubuntu) |
|---|---|---|
| CUDA Toolkit 11+ | GPU compute | nvidia installer |
| GMP | big-number math | `libgmp-dev` |
| OpenCL (optional) | portable GPU | `ocl-icd-opencl-dev` |
| OpenMP (optional) | CPU threading | `libomp-dev` |

## usage

```bash
# default: cuda integration, 1 billion intervals
./pi

# compute 1 million digits of pi
./pi --digits 1000000

# binary splitting on cpu (faster for large counts)
./pi --mode cpu --digits 1000000 --algo binsplit

# opencl mode
./pi --mode opencl --device 0

# cpu mode with specific cores
./pi --mode cpu --cores 0,1,2,3

# save digits to file without terminal spam
./pi --digits 1000000 --no-print --output my_pi.txt
```

### options

| flag | description | default |
|---|---|---|
| `N` | intervals for integration | 1,000,000,000 |
| `--mode` | backend: `cuda`, `opencl`, `cpu` | `cuda` |
| `--device N` | GPU index | `0` |
| `--cores LIST` | CPU cores to pin | all |
| `--digits D` | compute D decimal digits | skip |
| `--algo` | `chudnovsky` or `binsplit` | `chudnovsky` |
| `--output FILE` | digit output file | `pi.txt` |
| `--print` / `--no-print` | show digits in terminal | print |
| `--benchmark` | full benchmark suite | |
| `--benchmark-quick` | quick benchmark (10k + 100k digits) | |
| `--benchmark-gpu` | integration throughput only | |
| `--benchmark-cpu` | CPU digit computation only | |

### benchmark

```bash
# full suite: integration + digits at 10k/100k/1M/10M
./pi --benchmark

# quick: integration + digits at 10k/100k only
./pi --benchmark-quick

# gpu only: integration throughput
./pi --benchmark-gpu

# cpu only: chudnovsky + binsplit digit computation
./pi --benchmark-cpu
```

## gpu support

| architecture | GPUs | SM |
|---|---|---|
| Maxwell | GTX 9xx | 50/52 |
| Pascal | GTX 10xx, P100 | 60/61 |
| Volta | V100, Titan V | 70 |
| Turing | RTX 20xx, GTX 16xx | 75 |
| Ampere | RTX 30xx, A100 | 80/86 |
| Ada Lovelace | RTX 40xx | 89 |
| Hopper | H100 | 90 |
| Blackwell | B100/B200 | 100 |

## license

MIT - do whatever

## made by

[Tymbark7372](https://github.com/Tymbark7372)
