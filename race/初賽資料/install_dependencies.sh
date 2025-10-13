#!/bin/bash

# Installation script for A100 GPU environment
# CUDA 11.8 version

echo "Installing dependencies for A100 optimized model..."

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric and dependencies
pip install torch-geometric==2.4.0
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install tree models
pip install lightgbm==4.1.0 --install-option=--gpu
pip install xgboost==2.0.3
pip install catboost==1.2.2

# Install TabNet
pip install pytorch-tabnet==4.1.0

# Install other dependencies
pip install pandas numpy scikit-learn tqdm joblib scipy matplotlib seaborn

echo "Installation complete!"
echo "Verifying CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
