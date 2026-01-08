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
# Find all library directories in the 'nvidia' python package
NVIDIA_LIBS=$(python3 -c "import os, nvidia; nvidia_path = os.path.dirname(nvidia.__file__); 
lib_paths = []
for root, dirs, files in os.walk(nvidia_path):
    if 'lib' in dirs:
        lib_paths.append(os.path.join(root, 'lib'))
print(':'.join(lib_paths))" 2>/dev/null)

if [ ! -z "$NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVIDIA_LIBS
    echo "✅ LD_LIBRARY_PATH updated."
    
    # 3.1 Aggressive Symlinking: Provide .so.8 names for .so.9 files
    echo "   🔗 Creating .so.8 symlinks for CUDNN 9 compatibility..."
    python3 -c "import os; libs = '$NVIDIA_LIBS'.split(':'); 
for lib_dir in libs:
    if not os.path.exists(lib_dir): continue
    for f in os.listdir(lib_dir):
        if f.endswith('.so.9'):
            target = os.path.join(lib_dir, f.replace('.so.9', '.so.8'))
            if not os.path.exists(target):
                os.symlink(os.path.join(lib_dir, f), target)
                print(f'      linked {f} -> {os.path.basename(target)}')
" 2>/dev/null
    
    # Additional check for the main libcudnn.so.8
    CUDNN_8_PATH=$(find $NVIDIA_LIBS -name "libcudnn.so.8" | head -n 1)
    if [ ! -z "$CUDNN_8_PATH" ]; then
        echo "   ✅ Verified libcudnn.so.8 is available."
    else
        echo "   ⚠ Warning: libcudnn.so.8 still not found after symlinking."
    fi
fi

# 4. Launch Fooocus
echo "Step 3: Launching Fooocus..."
# --listen: allows external connections
# --port 7865: matches RunPod default exposed port
# --share: creates a gradio.live link
python Fooocus/launch.py --listen --port 7865 --share
