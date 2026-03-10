# TTT-MoE (The repository is currently being checked and refined.)

Official repository for **MoE-Enhanced-TTT: Advancing Medical Image Segmentation**. This code is based on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [U-Mamba](https://github.com/bowang-lab/U-Mamba), and [TTT-Unet](https://github.com/rongzhou7/TTT-Unet), and serves as the official implementation of our paper.

## Architecture

<img src="assets/TTT Framework.png" alt="TTT-MoE Architecture" width="700"/>

## Self-Supervised Learning in TTT-MoE

<img src="assets/Self Supervised Learning.png" alt="Self-Supervised Learning in TTT-MoE" width="700"/>

## Installation 

Requirements: `Ubuntu 20.04`, `CUDA 11.8`

1. Create a virtual environment: `conda create -n ttt-moe python=3.10 -y` and `conda activate ttt-moe `
2. Install [Pytorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4) 2.0.1: `pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118`
3. Install [Mamba](https://github.com/state-spaces/mamba): `pip install causal-conv1d>=1.2.0` and `pip install mamba-ssm --no-cache-dir`
4. `cd U-Mamba/umamba` and run `pip install -e .`


sanity test: Enter python command-line interface and run

```bash
import torch
import mamba_ssm
```

### Sanity Test

```python
import torch
import mamba_ssm
import causal_conv1d
```

## Model Training

Download dataset [here](https://drive.google.com/drive/folders/1DmyIye4Gc9wwaA7MVKFVi-bWD2qQb-qN?usp=sharing) and put them into the `data` folder. TTT-MoE is built on the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework. If you want to train on your own dataset, please follow this [guideline](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) to prepare the dataset.

### Preprocessing

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

### Train 2D Models

```bash
nnUNetv2_train DATASET_ID 2d all -tr ttt_moe
```

### Train 3D Models

```bash
nnUNetv2_train DATASET_ID 3d_fullres all -tr ttt_moe
```

## Inference

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIGURATION -f all -tr ttt_moe --disable_tta
```

> `CONFIGURATION` can be `2d` or `3d_fullres` for 2D and 3D models, respectively.

## Remarks

1. **Path settings**

   The default data directory is preset to `TTT-MoE/data`. Users with existing nnUNet setups who wish to use alternative directories for `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` can adjust these paths in `uttt/nnunetv2/paths.py`:

   ```python
   base = '/home/user_name/Documents/TTT-MoE/data'
   nnUNet_raw = join(base, 'nnUNet_raw')
   nnUNet_preprocessed = join(base, 'nnUNet_preprocessed')
   nnUNet_results = join(base, 'nnUNet_results')
   ```

2. **AMP and NaN issues**

   AMP (Automatic Mixed Precision) could lead to NaN values in the Mamba module. If you encounter NaN during training, consider disabling AMP.

3. **Dependency compatibility**

   PyTorch 2.1.0 requires `numpy<2` and `transformers<4.40`. If you encounter import errors after `pip install -e .`, run:
   ```bash
   pip install "numpy<2" "transformers<4.40"
   ```

## Paper
```
@INPROCEEDINGS{chen2025moe,
  author={Chen, Yuming and Wu, Qi and Xie, Yutong},
  booktitle={2025 International Conference on Digital Image Computing: Techniques and Applications (DICTA)}, 
  title={MoE-Enhanced-TTT: Advancing Medical Image Segmentation}, 
  year={2025},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/DICTA68720.2025.11302485}}

```

## Acknowledgements

This project is based on **nnU-Net**, **U-Mamba**, and **TTT-Unet**. We acknowledge all the authors of the employed public datasets, as well as the authors of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [U-Mamba](https://github.com/bowang-lab/U-Mamba), [Mamba](https://github.com/state-spaces/mamba), and [TTT-Unet](https://github.com/rongzhou7/TTT-Unet) for making their valuable code publicly available.
