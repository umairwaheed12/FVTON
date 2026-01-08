#!/bin/bash

# RunPod Setup Script for FVTON (Virtual Try-On)
# This script automates cloning, dependencies, and launching.

echo "==========================================="
echo "   FVTON RUNPOD SETUP & LAUNCH SCRIPT      "
echo "==========================================="
echo "TIP: To keep this running in the background, run this script inside a 'tmux' session."
echo "     Usage: tmux new -s fvton"
echo "            bash runpod_setup.sh"
echo "            (Press Ctrl+B then D to detach)"
echo ""

# 1. Check if we are already in the project directory
if [[ "$PWD" == *"/FVTON"* ]]; then
    echo "Already in FVTON directory."
else
    if [ -d "FVTON" ]; then
        echo "FVTON directory exists, entering..."
        cd FVTON
    else
        echo "Cloning FVTON repository..."
        git clone https://github.com/umairwaheed12/FVTON.git
        cd FVTON
    fi
fi

# 2. Run the environment setup and model download
echo "Step 1: Setting up environment and downloading models..."
python Fooocus/modules/download_small_models.py

# 3. Fix ONNX Runtime GPU (CUDNN) library links
echo "Step 2: Linking CUDNN libraries for GPU acceleration..."
CUDNN_PATH=$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])" 2>/dev/null)
if [ ! -z "$CUDNN_PATH" ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDNN_PATH/lib
    echo "✅ LD_LIBRARY_PATH updated with CUDNN."
fi

# 4. Launch Fooocus
echo "Step 3: Launching Fooocus..."
# --listen: allows external connections
# --port 7865: matches RunPod default exposed port
# --share: creates a gradio.live link
python Fooocus/launch.py --listen --port 7865 --share
