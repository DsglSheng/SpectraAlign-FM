# SpectraAlign-FM

SpectraAlign-FM learns transferable spectral–spatial representations from hyperspectral remote sensing imagery. It routes heterogeneous bands using wavelength, bandwidth, validity, and sample metadata, aligns patch features with sensor-specific ground spectrum prototypes, and reconstructs masked inputs across spectral, spatial, and frequency domains.

The joint pretraining recipe uses HyperGlobal-DZ1. It covers EO-1 Hyperion, GF-5 AHSI, and DZ1 VNIR and SWIR imagery. The Python package is imported as `aware_open`.

## Install

Use Python 3.10 or newer from the repository root. For the CUDA 12.8 environment used by the training launchers, install the PyTorch build first.

```bash
python -m pip install -r requirements-torch-cu128.txt
python -m pip install -e ".[experiments]"
```

For CPU inspection and tests, install a CPU build of PyTorch for your platform before installing the project.

## Verify the installation

```bash
python -m unittest discover -s tests -v
python pretraining/scripts/train_foundation_dz1.py --help
python downstream/scripts/run_task_from_config.py --list
python downstream/scripts/run_task_from_config.py --check --allow-missing-checkpoint
```

## Repository layout

| Path | Contents |
| --- | --- |
| `aware_open/` | Band and sample metadata, encoder, prototype alignment, reconstruction, and downstream adapters |
| `pretraining/` | DZ1 data preparation, joint manifests, prototype banks, and the two-stage joint training recipe |
| `scripts/` | Dataset preparation, the EO-1/GF-5 training entry point, and downstream fine-tuning scripts |
| `downstream/` | Task registry, dataset-specific configurations, and launchers |
| `benchmarks/inversion/` | Hyperspectral regression and quantitative inversion experiments |
| `examples/paper_case_studies/` | Oil-spill detection, Cuprite geology, and feature visualization |
| `experiments/` | Satellite-to-UAV transfer and Pavia University feature analysis |
| `tests/` | Data, model, and training unit tests |

See the [joint pretraining guide](pretraining/README.md), the [downstream guide](downstream/README.md), and the [EO-1/GF-5 recipe](docs/eo1_gf5_pretraining.md) for the commands and inputs for each workflow.

Training and evaluation use local datasets, prototype banks, and checkpoints. Set `SPECTRAALIGN_DATA_ROOT` to the absolute path to those assets or use the default `data/` directory at the repository root. The supplementary comparison scripts also need the external baselines, their checkpoints, and paths appropriate to your environment.

## Third-party code

The vendored `dinov3/` modules are distributed under the [included DINOv3 License](DINOV3_LICENSE.md).
