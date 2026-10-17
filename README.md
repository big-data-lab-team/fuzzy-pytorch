# Fuzzy PyTorch: Rapid Numerical Variability Evaluation for Deep Learning Models

> [!IMPORTANT]
> The latest version of Fuzzy-Pytorch can be found in the [verificarlo/fuzzy repository](https://github.com/verificarlo/fuzzy#fuzzy-pytorch).

This repository contains the experimental code and reproducibility artifacts for the paper "Fuzzy PyTorch: Rapid Numerical Variability Evaluation for Deep Learning Models".
  
## Overview

Fuzzy PyTorch provides tools and methodologies for evaluating numerical variability and floating-point precision effects in deep learning models. This repository contains three main experimental evaluations comparing various floating-point error analysis tools:

**Key Capabilities:**
- Analyze numerical variability effects on deep learning models
- Implement stochastic rounding (SR) and up-down (UD) rounding modes
- Minimal code modifications required


## Prerequisites

> **⚠️ Platform Support:** Fuzzy PyTorch is currently designed for Linux systems. macOS and Windows are not officially supported. All build instructions, container recipes, and command-line examples assume a Linux environment and require a local build and compilation.

- **Linux operating system**
- **Apptainer/Singularity**: Container runtime for reproducible execution
- **SLURM** (optional): For automated batch job execution
- **Python 3.x**: For analysis notebooks
- **Jupyter**: For running analysis notebooks

## Quick Start

Fuzzy PyTorch integrates stochastic arithmetic directly into PyTorch's execution pipeline by combining the Verificarlo compiler with the PRISM backend. This enables rapid numerical variability analysis without modifying your model code or architecture.

Before diving into experiments, familiarize yourself with Verificarlo's architecture and PRISM backend:

- **Verificarlo Repository:** https://github.com/verificarlo/verificarlo
  - Complete documentation on numerical backends and usage patterns
  
- **PRISM Backend Documentation:** https://github.com/verificarlo/verificarlo/blob/master/doc/02-Backends.md#prism-backend
  - Usage instructions, limitations, and configuration options

### Applying Fuzzy PyTorch to Your Own DL Use Case

**1. Build a Fuzzy PyTorch Base Image**

Choose the rounding mode you want to use:

* SR — stochastic rounding (recommended for realistic large scale experiments)

* UD — up-down rounding (faster for small experiments, but does not scale)

Build the appropriate image by selecting the corresponding Dockerfile (ex: [Dockerfile-pytorch-sr](https://github.com/big-data-lab-team/fuzzy-pytorch/blob/main/containers/Dockerfile-pytorch-sr)) in the `containers/` directory.

**2. Extend the Base Image for Your Use Case**

You can use the base image directly if it includes all required packages.
Otherwise, create a custom Docker image using the Fuzzy PyTorch base as your parent.

> **Important**: When installing Python packages that depend on PyTorch or BLAS/LAPACK, use `--no-deps` to avoid overwriting the instrumented Fuzzy-PyTorch build.

Example: Fuzzy PyTorch SR with Transformers Support
```
FROM fuzzy-pytorch:sr

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools

# Install required libraries (avoid overwriting PyTorch/BLAS/LAPACK)
RUN pip install transformers==4.37.2 numpy==1.24.1 --no-deps
RUN pip install joblib==1.4.2 tokenizers==0.15.0 huggingface_hub==0.29.3 tqdm==4.67.1 \
               regex==2024.11.6 safetensors==0.5.3 hyperpyyaml==1.2.2 ruamel.yaml==0.18.10 --no-deps

ENTRYPOINT ["/bin/bash"]
```

**3. Run Your Model with Fuzzy PyTorch**

Execute your training or inference script inside the container. \
You may also convert the container to Apptainer/Singularity if needed for cluster environments. \
If your model produces floating-point outputs, you can optionally use the [significant_digits](https://github.com/verificarlo/significantdigits
) package to quantify numerical variability using the significant bits metric.


### Reproducing Experiments From Paper
**Build containers**:
   ```bash
   cd containers
   ./build.sh
   ```

**Run experiments**: Navigate to each experiment directory and follow the instructions in their respective README files.

**Generate figures**: Execute the Jupyter notebooks in each experiment directory to reproduce the analysis and figures.

## Tools Evaluated

The experiments compare the following floating-point error analysis tools:

- **IEEE**: Standard IEEE 754 floating-point arithmetic
- **PRISM**: Precision analysis with stochastic rounding variants (SR, UD)
- **Verrou**: Monte Carlo Arithmetic with CESTAC and SR modes
- **CADNA/CESTAC**: Control of Accuracy and Debugging for Numerical Applications
- **Verificarlo**: Monte Carlo Arithmetic with Random Rounding (MCA RR)
- **FM SR**: Fast Math Stochastic Rounding

## Tested Use Cases
- **Deep Learning Benchmarks**: Performance evaluation on real-world ML models (MNIST, FastSurfer, WavLM)
- **Harmonic Series Analysis**: Numerical accuracy assessment using mathematical series computations
- **NAS Parallel Benchmarks**: Performance overhead analysis using standard HPC benchmarks

## Repository Structure

```
├── containers/           # Container definitions for reproducible experiments
│   ├── Dockerfile-NPB   # NPB benchmarks environment
│   ├── Dockerfile-harmonic  # Harmonic series environment  
│   ├── Dockerfile-tools # Deep learning environment
│   ├── Dockerfile-fastsurfer  # FastSurfer with fuzzy Pytorch container definition (mode needs to be set)
│   ├── Dockerfile-freesurfer  # FreeSurfer container definition
│   ├── Dockerfile-pytorch-ud # MNIST/Base fuzzy Pytorch UD container definition
│   ├── Dockerfile-pytorch-sr # MNIST/Base fuzzy Pytorch SR container definition
│   ├── Dockerfile-wavlm # WavLM with fuzzy Pytorch container definition
│   └── NPB/             # NPB source and build scripts
│   └── harmonic/             # Harmonic source and build scripts
│   └── tools/             # Additional scripts
├── experiments/         # Experimental evaluations
│   ├── DL/              # Deep learning performance evaluation
│   │   ├── Fuzzy_PyTorch.ipynb            # Main notebook with MNIST and FastSurfer results
│   │   ├── FastSurfer_Use_Case/       # Experiments with FastSurfer models
│   │   │   ├── allsub_fast.txt        # List run commands per subject for parallelization
│   │   │   ├── fastsurfer_embeddings.pdf    # Embeddings visualization
│   │   │   ├── ieee_subjects.txt      # IEEE subjects list
│   │   │   ├── run_fuzzy.sh           # Run fuzzy experiments
│   │   │   ├── run_verrou.sh          # Run Verrou experiment
│   │   │   ├── subjects.txt           # All subjects list
│   │   │   └── verrou_sr_min_dice_scores.csv   # Minimum Dice scores from Verrou SR FastSurfer inference
│   │   │   └── verrou_ud_min_dice_scores.csv   # Minimum Dice scores from Verrou CESTAC FastSurfer inference
│   │   │   └── fuzzy_sr_min_dice_scores.csv   # Minimum Dice scores from Fuzzy SR FastSurfer inference
│   │   │   └── fuzzy_ud_min_dice_scores.csv   # Minimum Dice scores from Fuzzy UD FastSurfer inference
│   │   │   └── ieee_min_dice_scores.csv   # Minimum Dice scores from IEEE default FastSurfer inference
│   │   ├── MNIST_Use_Case/            # Experiments with MNIST dataset
│   │   │   ├── mnist_test.py          # MNIST testing script
│   │   │   ├── run_embedding.sh       # Run embedding experiments
│   │   │   └── run_mnist.sh           # Run MNIST experiments
│   │   └── WavLM_Use_Case/            # Experiments with WavLM speech model
│   │   │   ├── WavLM.ipynb            # Main WavLM notebook
│   │   │   ├── inference_only.py      # Script for inference
│   │   │   ├── run_iter.sh            # Iterative run script
│   │   │   ├── run_model.sh           # Model run script
│   │   │   ├── run_verrou.sh          # Run Verrou instrumentation
│   │   │   ├── train.yaml             # Training configuration
│   │   ├── figures             
│   ├── NPB/             # NAS Parallel Benchmarks analysis
│   └── harmonics/       # Harmonic series numerical analysis
└── README.md           # This file
```
Building `Dockerfile-pytorch-ud` or `Dockerfile-pytorch-sr` requires the [Fuzzy](https://github.com/verificarlo/fuzzy) repository and replacing [fuzzy/docker/resources/pytorch/pytorch-vfc-exclude.txt](https://github.com/verificarlo/fuzzy/blob/master/docker/resources/pytorch/pytorch-vfc-exclude.txt) with the version found in `tools`

## Experiments

### 1. Deep Learning Performance ([experiments/DL/](experiments/DL/))
Evaluates the runtime overhead of floating-point error analysis tools on deep learning workloads including MNIST classification, FastSurfer brain segmentation, and WavLM speech processing.

### 2. Harmonic Series Analysis ([experiments/harmonics/](experiments/harmonics/))
Assesses numerical accuracy and convergence properties of harmonic series computations across different floating-point precision analysis methods.

### 3. NAS Parallel Benchmarks ([experiments/NPB/](experiments/NPB/))
Measures performance overhead of various floating-point error analysis tools using the standard NAS Parallel Benchmarks suite (BT, CG, EP, FT, LU, MG, SP).

## Citation

If you use this work, please cite:
```
Fuzzy PyTorch: Rapid Numerical Variability Evaluation for Deep Learning Models
```

## License

See individual component licenses in their respective directories.
