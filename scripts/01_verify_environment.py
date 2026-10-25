# -*- coding: utf-8 -*-
"""
Script 01: Verify Environment & Hardware Acceleration (GPU Check)
Checks CUDA availability, PyTorch GPU, XGBoost GPU capability, and all required Gap 3 dependencies.
"""

import sys

def verify_all():
    print("=" * 60)
    print("EEG Environment & GPU Verification (Gap 3 Setup)")
    print("=" * 60)
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version:    {sys.version.split()[0]}")
    print("-" * 60)

    # 1. Check PyTorch & CUDA GPU
    gpu_available = False
    try:
        import torch
        print(f"[OK] PyTorch Version: {torch.__version__}")
        if torch.cuda.is_available():
            gpu_available = True
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[OK] CUDA Available: YES (Device Count: {device_count})")
            print(f"[OK] Primary GPU:    {device_name} ({vram_gb:.2f} GB VRAM)")
        else:
            print("[WARN] CUDA Not Available in PyTorch (Running on CPU)")
    except ImportError as e:
        print(f"[ERROR] PyTorch Import Failed: {e}")

    # 2. Check XGBoost & GPU
    try:
        import xgboost as xgb
        print(f"[OK] XGBoost Version: {xgb.__version__}")
        try:
            # Quick GPU test for XGBoost
            dtrain = xgb.DMatrix([[1.0, 2.0], [3.0, 4.0]], label=[0, 1])
            params = {'tree_method': 'hist', 'device': 'cuda' if gpu_available else 'cpu'}
            bst = xgb.train(params, dtrain, num_boost_round=1)
            target_device = "GPU (cuda)" if gpu_available else "CPU"
            print(f"[OK] XGBoost Device Acceleration: Verified on {target_device}")
        except Exception as e:
            print(f"[WARN] XGBoost GPU test note: {e}")
    except ImportError as e:
        print(f"[ERROR] XGBoost Import Failed: {e}")

    # 3. Check Core ML Packages
    packages = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-Learn"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("tqdm", "TQDM"),
        ("openml", "OpenML"),
        ("tensorflow", "TensorFlow")
    ]

    print("-" * 60)
    all_ok = True
    for mod_name, disp_name in packages:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, '__version__', 'Installed')
            print(f"[OK] {disp_name:<15}: {ver}")
        except ImportError as e:
            print(f"[MISSING] {disp_name:<15}: {e}")
            all_ok = False

    print("=" * 60)
    if all_ok:
        print("[SUCCESS] All environment requirements for Gap 3 are verified and ready!")
    else:
        print("[WARNING] Some packages are missing. Check above output.")
    print("=" * 60)

if __name__ == "__main__":
    verify_all()
