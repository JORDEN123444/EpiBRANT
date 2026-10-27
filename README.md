
# EpiBRANT: A Scale-Aligned Intracranial EEG Foundation Model

Official research repository for **EpiBRANT (Epilepsy Brain Representation with Adaptive Neural Transformers)**, an epilepsy-focused intracranial electroencephalography (iEEG) foundation model designed to learn transferable electrophysiological representations from heterogeneous intracranial recordings.

EpiBRANT combines scale-aligned time–frequency representations, temporal Transformer encoding, adaptive cross-channel attention, and self-supervised masked spectrogram modeling to support intracranial EEG representation learning and epilepsy-related downstream tasks.

## Overview

Intracranial EEG recordings provide valuable information about epileptic brain dynamics. However, differences in acquisition systems, sampling frequencies, electrode configurations, and recording characteristics present challenges for developing transferable neural representations.

EpiBRANT addresses these challenges through a self-supervised pretraining framework that learns representations from intracranial EEG recordings without requiring manual clinical annotations during pretraining.

The framework supports model pretraining, downstream fine-tuning, and evaluation across heterogeneous intracranial EEG datasets.

## Framework

The EpiBRANT pipeline comprises the following stages:

1. Dataset indexing and recording metadata preparation.
2. Signal preprocessing and temporal window extraction.
3. Time–frequency representation generation.
4. Scale-aligned spectral embedding.
5. Temporal Transformer encoding.
6. Adaptive cross-channel representation learning.
7. Self-supervised masked spectrogram reconstruction.
8. Downstream fine-tuning and evaluation.

## Repository Structure

The repository is organized around the following components:

- `models/` — Model architecture and representation-learning components.
- `preprocessing/` — Signal preprocessing and time–frequency representation generation.
- `pretraining/` — Self-supervised pretraining procedures.
- `fine_tuning_tasks/` — Downstream task-specific training and evaluation.
- `evaluation/` — Evaluation metrics and analysis utilities.
- `configs/` — Model and experiment configurations.
- `datasets/` — Dataset-loading utilities and documentation.
- `examples/` — Example usage and model execution.

*Note: Directory names should be aligned with the actual repository structure before publication.*

## Data Preparation

EpiBRANT uses heterogeneous intracranial EEG recordings from multiple data sources.

The data preparation pipeline organizes recording metadata, identifies valid signal segments, and prepares inputs for model training and evaluation. Dataset-specific preprocessing and loading procedures should be configured according to the characteristics and access requirements of each dataset.

Raw clinical recordings are not included in this repository. Researchers must obtain access to the relevant datasets through their respective providers and comply with applicable data-use agreements, ethical approvals, and privacy requirements.

## Signal Preprocessing and Time–Frequency Representation

The preprocessing pipeline prepares intracranial EEG recordings for scale-aligned representation learning.

The workflow includes signal preprocessing, temporal segmentation, and time–frequency transformation. Spectrogram representations are generated using the short-time Fourier transform (STFT), with the specified windowing and frequency-alignment settings documented in the experiment configurations.

Exact preprocessing parameters, including sampling frequency, filtering, window duration, patch duration, and STFT settings, should be reported consistently with the implementation used in the manuscript.

## EpiBRANT Architecture

### Scale-alignment module

The scale-alignment module maps time–frequency representations into a shared embedding space to facilitate representation learning across heterogeneous intracranial recordings.

### Temporal Transformer encoder

The temporal encoder models sequential dependencies among temporal signal patches to learn representations of evolving neural activity.

### Adaptive cross-channel attention

The cross-channel attention module models relationships among intracranial recording channels, supporting the representation of spatially distributed electrophysiological activity.

### Self-supervised reconstruction

During pretraining, selected spectrogram elements are masked, and the model learns to reconstruct the masked information. This objective supports representation learning without requiring manual clinical labels.

## Model Pretraining

The pretraining pipeline comprises:

1. Loading and preparing intracranial EEG recordings.
2. Extracting temporal windows and generating time–frequency representations.
3. Constructing scale-aligned input embeddings.
4. Applying the self-supervised masking strategy.
5. Encoding temporal and cross-channel dependencies.
6. Computing the reconstruction objective.
7. Updating model parameters using the configured optimizer.

Training parameters, model configurations, and execution commands should be provided with the corresponding implementation files.

## Downstream Fine-Tuning and Evaluation

The pretrained EpiBRANT representation can be evaluated on supported epilepsy-related downstream tasks.

These include seizure-related classification, sleep–wake state classification, signal forecasting, signal imputation, and seizure-onset-zone localization, where implemented and supported by the corresponding datasets.

The repository should provide task-specific configurations, evaluation procedures, and metric calculations consistent with the experiments reported in the manuscript.

## Installation and Requirements

The software environment should specify the Python version, PyTorch version, relevant scientific-computing libraries, and GPU/CUDA requirements used in the experiments.

Install the dependencies listed in the repository's `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Additional environment configuration may be required depending on the hardware and dataset-loading procedures.

## Reproducibility

The repository is intended to support transparent evaluation and reproducibility of the EpiBRANT framework through:

- Model architecture and training code.
- Preprocessing and dataset-loading utilities.
- Experiment configurations.
- Downstream fine-tuning and evaluation procedures.
- Documentation of software and hardware requirements.

Reproduction of the reported results depends on access to the required datasets, the corresponding experiment configurations, and compatible computational resources.

## Pretrained Model

The pretrained EpiBRANT model is intended to be made available through Hugging Face:

[Hugging Face model repository](https://huggingface.co/ijaz0310/EpiBRANT)

Model availability, checkpoint versions, usage instructions, and applicable terms should be documented in the model repository.

## Data Availability

Raw clinical recordings are not distributed with this code repository. Access to individual datasets is subject to the policies and conditions established by the respective data providers.

## Code Availability

The EpiBRANT implementation is maintained in this GitHub repository. The repository's release and version information should be used to identify the code corresponding to the published manuscript.

## Citation

If you use this implementation, please cite the EpiBRANT manuscript:

**Ahmad, I. et al.** *EpiBRANT: A Scale-Aligned Intracranial Foundation Model for Clinical Epilepsy Analysis.*

Add the final journal citation and DOI when available.

## Acknowledgements

EpiBRANT builds on advances in self-supervised learning, Transformer-based neural representations, and foundation-model development for electrophysiological signals. Relevant prior methods should be cited in the manuscript and repository documentation.
