# Ship Collision Avoidance Target Model

**English** | [日本語](README_ja.md)

## Overview

This repository provides training code, trained models, and simulation code for an **Interactive Target Ship Behavior Model** that represents target-ship behavior probabilistically during collision avoidance.

Based on the relative geometry and motion states of two vessels, the model estimates which side of the own ship a target ship will pass and where it will pass. It represents multiple possible behaviors under the same encounter conditions and can be used in repeated simulations to assess safety while accounting for uncertainty and mutual interactions between vessels.

The code has been organized for understanding and reusing the research methods. **The original training dataset is not publicly available, but the trained models are provided.**

## Repository Contents

- **Model training code:** Training a passing-side classifier and probabilistic passing-position models
- **Trained models:** One classifier and separate log-normal distribution models for starboard and port passing
- **Simulation code:** Encounter simulations and repeated trials using the integrated model
- **Scripts and utilities:** Command-line workflows for data generation, training, and simulation

## Model Overview

The model uses encounter geometry, including the Predicted Point of Collision (PPC), and vessel motion states to predict two aspects of a future passing position:

1. **Passing Side:** A classification model estimates the probabilities of passing on the starboard and port sides of the own ship.
2. **Passing Position:** Separate log-normal distribution models represent the X coordinate of the future predicted collision point for each passing side. Refer to the implementation and data documentation for coordinate and sign conventions.

The three models are combined using `ModelIntegrator` and then used in the simulation.

## Main Workflow

1. Prepare `data/input_data.npy` and `data/output_data.npy` from the source dataset.
2. Train the passing-side classifier, `Classification.h5`.
3. Train the starboard and port passing-position distribution models, `LogNormStb.h5` and `LogNormPrt.h5`.
4. Combine the three trained models using `ModelIntegrator`.
5. Use the integrated model as the target-ship behavior model in a representative collision-avoidance simulation corresponding to `AgentSimulation11_ins1`. Repeat trials as needed and summarize safety-related measures such as passing distance.

## Repository Layout

```text
src/ship_collision_model/
  target_model.py       # Classification and log-normal model integration
  simulation.py         # Encounter simulation and repeated trials
  ships.py              # Ship class and maneuver decision logic
  geometry.py           # Navigation, CPA, and predicted collision point utilities
  losses.py             # Keras NLL losses and evaluation metrics
  evaluation.py         # Gamma vs. log-normal comparison utilities
  sample_data.py        # Shape-compatible synthetic data generator

scripts/
  generate_dummy_data.py
  train_classifier.py
  train_passage_distribution.py
  run_simulation_demo.py

data/
  README.md

models/
  README.md

outputs/
  README.md

```

## Required Data and Models

The original research dataset is excluded due to data usage restrictions. The main expected files are:

| File | Description |
| --- | --- |
| `data/input_data.npy` | Model input data; shape `(N, 6)` |
| `data/output_data.npy` | Training targets; shape `(N, 2)` |
| `models/Classification.h5` | Trained passing-side classifier |
| `models/LogNormStb.h5` | Trained starboard-side distribution model |
| `models/LogNormPrt.h5` | Trained port-side distribution model |

See [data/README.md](data/README.md) and [models/README.md](models/README.md) for details on file formats and locations. Because the original dataset is unavailable, training cannot be reproduced under exactly the same conditions as in the research. The trained models are intended to support inference and simulation without the original dataset.

## Quick Smoke Test

To check the expected array shapes and basic code execution without the original dataset, generate synthetic dummy data:

```bash
pip install -e .
python scripts/generate_dummy_data.py
```

**Note:** Synthetic dummy data is only for checking data structures and code execution. It must not be used to reproduce research results or evaluate model accuracy. The command above does not run a simulation with the trained models.

## Running Workflows

With `data/input_data.npy` and `data/output_data.npy` in place, train the models with:

```bash
python scripts/train_classifier.py
python scripts/train_passage_distribution.py
```

With `data/input_data.npy` and the three trained model files in `models/`, run the representative simulation with:

```bash
python scripts/run_simulation_demo.py
```

The original notebooks remain in `legacy/` as historical materials and are not part of the supported workflow.

## Limitations and Disclaimer

This repository is provided for research and educational purposes. It does not guarantee collision-avoidance performance or safety in real-world ship operations. Model predictions depend on the training dataset and should be interpreted carefully for encounter conditions that are not adequately represented in that dataset.

## License

The source code and trained model weights provided in this repository are released under the MIT License.

The original dataset used for training was provided by a third party and is not included in this repository. The dataset is not covered by the license of this repository.

See the LICENSE file for details.
