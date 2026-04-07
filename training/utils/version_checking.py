import sys
import torch
import mlflow
import nibabel as nib

def check_environment():
    print("="*50)
    print("Environment Check for AI Project")
    print("="*50)

    # 1. Python Version
    print(f"Python Version: {sys.version.split()[0]}")

    # 2. PyTorch & CUDA (สำคัญมากสำหรับ H-100)
    print(f"PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        # ตรวจสอบ Memory ตามสเปกเครื่องที่คุณใช้ (10GB) 
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Total GPU Memory: {total_mem:.2f} GB")

    # 3. MLflow Version [cite: 137]
    print(f"MLflow Version: {mlflow.__version__}")

    # 4. NiBabel Version (สำหรับจัดการไฟล์ NIFTI) [cite: 138]
    print(f"NiBabel Version: {nib.__version__}")
    print("="*50)

if __name__ == "__main__":
    check_environment()