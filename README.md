# Enhancing U-Net based Model for Small Tumor Segmentation in Brain MRI

## About This Repository
This repository is for CP413706 Research Methodology and CP414781 Research Project in Artificial Intelligence I at Khon Kaen University.

* **Goal:** Modify U-Net based model to improve segmentation result on BraTS 2024 dataset.
* **Tech Stack:** Python 3.10, PyTorch, MLFlow, LenovoLiCO H-100.
* **Key Features:** Automatic logging, checkpoint saving, and easy to resume training on LenovoLiCO.

## How to run
### 1. Install dependency with conda virtual environment
```bash
conda create -n brain_mri python=3.10
conda activate brain_mri
pip install mlflow
```

### 2. Run MLFlow server and create .env
You must run mlflow on your computer at the same local network (KKU network). This example run MLFlow server command is for linux device.
```bash
nohup mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns --serve-artifacts > mlflow.log 2>&1 &
```

You also must allowed port 5000 to let other device can access. This example is for linux device.
```bash
sudo ufw enable
sudo ufw allow 5000
```

Create .env file and change ip and port to your device (MLFlow server)
```bash
cp training/.env.example .env
```

### 3. Prepare LenovoLiCO H-100 environment
Copy training folder and upload into LenovoLiCO H-100 and create folder named 'BraTS-Datasets' for upload dataset under same root folder as you upload training folder.

Then upload data into 'BraTS-Datasets' folder.
```bash
root
  |--training
  |--BraTS-Datasets
```

### 4. Training and run arguments
Select PyTorch job template on LenovoLiCO and select working directory as root folder that you create on step 3.
```bash
root # <--Working directory on step 3
```

select run script
```bash
training/training.py <model_name> <other_options>
```

#### Configuration Arguments

The `training.py` script uses `argparse` to handle configurations. Below are the available command-line arguments:

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_name` | string | **Required** | Architecture to use (e.g., `u_net`, `bipyramid_se_di`, etc.). |
| `--data_dir` | string | `BraTS-96file-20per-train` | Path to the root directory of the dataset. |
| `--batch_size` | int | `16` | Batch size for the DataLoader. |
| `--se_reduction`| int | `16` | Squeeze and Excitation reduction rate. |
| `--dilation_rate`| list | `[1, 2, 1, 2]` | Dilation rates for the 4 U-Net layers. |
| `--lr` | float | `4e-4` | Learning rate for the optimizer. |
| `--epochs` | int | `25` | Total number of training epochs. |
| `--val_split` | float | `0.25` | Fraction of data used for validation. |
| `--num_classes` | int | `4` | Number of output segmentation classes. |
| `--resume` | flag | `False` | Use this flag to resume from the last checkpoint. |
| `--patience` | int | `5` | Epochs to wait for improvement before early stopping. |

---