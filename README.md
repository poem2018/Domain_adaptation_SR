# Domain Adaptation for Super-Resolution (SR)

## Overview

This repository focuses on domain adaptation in super-resolution tasks using a modified SR3 model architecture. The project involves:

- **Pretraining on the COCO dataset**.
- **Domain adaptation to IXI and fastMRI datasets**.
- **Utilizing ControlNet for fine-tuning**.

## File Structure

- **`config/`**: Contains training and validation parameter configurations.
  - `coco_128_256.json`
  - `IXI_64_128.json`
  - `ixi_fastmri.json`
  - `fastmri_128_256.json` (includes most of the test experiment configurations)
  - These JSON files represent configurations for:
    - COCO pretraining.
    - COCO-IXI (ci).
    - IXI-fastMRI (cif) experiments.
- **`data/`**: Contains data preprocessing scripts and data loader definitions.
  - `prepare_data_vlr.py`: Script for data preprocessing (generating LR and HR images).
  - `LRHR_dataset.py`: Definition of the data loader.
- **`model/`** and **`model/sr3_modules/`**: Define the model structure and modules.
- **`*.swb` files**: Scripts used to submit running jobs on the Delta GPU system.
- **`sr.py`**: The main script for training and inference.

## Data Preprocessing

- **Generating LR and HR Images**:
  - Use `prepare_data_vlr.py` to generate Low-Resolution (LR) and High-Resolution (HR) image pairs for training:
    ```bash
    python data/prepare_data_vlr.py
    ```
- **Data Loader Definition**:
  - The data loader is defined in `LRHR_dataset.py`.

## Training

- **Starting Training**:
  - Run the training script with the desired configuration:
    ```bash
    python sr.py -p train -c <config.json>
    ```
- **Training Stages**:
  - **COCO Pretraining**:
    - Initial training on the COCO dataset.
  - **COCO-IXI Domain Adaptation**:
    - Fine-tuning the model from COCO to IXI dataset.
    - ControlNet is incorporated during this stage to enhance fine-tuning.
  - **IXI-fastMRI Domain Adaptation**:
    - Further adapting the model from IXI to fastMRI dataset.

## ControlNet Fine-Tuning

- **Step 1: Modify the Model with ControlNet**:
  - Add ControlNet to the existing model using `tool_add_control.py`:
    ```bash
    python tool_add_control.py -i <checkpoint> -o ./finetune_model/ -c ./config.json
    ```
    - `<checkpoint>`: Path to the pre-trained model checkpoint.
    - `./finetune_model/`: Output directory for the modified model.
    - `./config.json`: Configuration file.
- **Step 2: Fine-Tune the Modified Model**:
  - Update the configuration file (`config.json`) with the following parameters:
    ```json
    "is_control": true,
    "orthogonal": false,
    "resume_state": "<new_ckpt>"
    ```
    - `"is_control": true` enables ControlNet.
    - `"resume_state": "<new_ckpt>"` specifies the path to the new checkpoint.
- **Submitting Jobs on Delta GPU System**:
  - Use the `*.swb` scripts to submit jobs:
    ```bash
    sbatch your_script.swb
    ```

## Inference

- **Running Inference**:
  - Perform inference using the validation phase:
    ```bash
    python sr.py -p val -c config/fastmri_128_256.json
    ```
    - `-p val`: Specifies the validation (inference) phase.
    - `-c config/fastmri_128_256.json`: Uses the configuration file for the fastMRI dataset.

## Getting Started

- **Prerequisites**:
  - Python 3.x
  - Required Python packages (install via `requirements.txt` if available)
  - Access to the Delta GPU system (for job submission scripts)

- **Preparing the Data**:
  - Ensure that the datasets are properly downloaded and placed in the designated directories as specified in the configuration files.

## Usage

- **Training**:
  - Replace `<config.json>` with the desired configuration file:
    ```bash
    python sr.py -p train -c config/coco_128_256.json
    ```
- **Inference**:
  - For inference on the fastMRI dataset:
    ```bash
    python sr.py -p val -c config/fastmri_128_256.json
    ```


