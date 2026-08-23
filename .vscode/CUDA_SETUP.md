# CUDA Setup for VSCode - Quick Reference

## ✅ What Was Configured

1. **settings.json** — CUDA environment variables
2. **tasks.json** — Build tasks for running Python with CUDA
3. **launch.json** — Debug configurations
4. **verify_cuda.py** — Script to verify CUDA is working

---

## 🚀 How to Use

### 1. **Verify CUDA Installation**
```
Ctrl + Shift + B  →  "Check CUDA & GPU Status"
```
Or run directly:
```powershell
python verify_cuda.py
```

### 2. **Run Python Scripts with CUDA**
```
Ctrl + Shift + B  →  "Run Python with CUDA"
```
Or in terminal:
```powershell
python your_script.py
```

### 3. **Run Jupyter Notebooks**
```
Ctrl + Shift + B  →  "Run Jupyter Notebook (CDCN Project)"
```

### 4. **Debug Python Code**
- Press `F5` or go to **Run → Start Debugging**
- Select "Python: Current File (with CUDA)"
- Set breakpoints with `F9`
- Step through code with `F10` (step over) / `F11` (step into)

### 5. **Monitor GPU While Running**
Open a new terminal (`Ctrl + ` `) and run:
```powershell
# Watch GPU usage in real-time (Ctrl+C to stop)
while ($true) { nvidia-smi; Start-Sleep -Seconds 2 }
```

---

## 🔍 Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Run task | `Ctrl + Shift + B` |
| Debug/Run | `F5` |
| Stop debugging | `Shift + F5` |
| Step over | `F10` |
| Step into | `F11` |
| Continue | `F5` (while debugging) |
| Breakpoint | `F9` |
| New terminal | `` Ctrl + ` `` |
| Select Python interpreter | `Ctrl + Shift + P` → "Python: Select Interpreter" |

---

## 🎯 Common Workflows

### Running CDCN Training Script
```powershell
# Terminal (Ctrl + `)
python CDCN_internship_project.ipynb
```

### Debugging a TensorFlow Script
1. Open your script (e.g., `train_model.py`)
2. Set breakpoints with `F9`
3. Press `F5` → Select "Python: Current File (with CUDA)"
4. Inspect variables in the Debug panel

### Testing GPU Computation
1. Run `verify_cuda.py` (Ctrl + Shift + B)
2. Check if "GPU computation works" message appears

---

## 📊 Environment Variables Set

These are automatically set in VSCode terminals:

- `CUDA_HOME`: `C:\Program Files\NVIDIA\CUDA\v13`
- `CUDA_PATH`: `C:\Program Files\NVIDIA\CUDA\v13`
- `CUDNN_HOME`: `C:\Program Files\NVIDIA\cuDNN`
- `CUDA_VISIBLE_DEVICES`: `0` (use GPU 0)
- `TF_CPP_MIN_LOG_LEVEL`: `1` (reduce TensorFlow log noise)

---

## ⚙️ Python Interpreter

VSCode is configured to use:
```
${workspaceFolder}/cdcn_gpu_env/Scripts/python.exe
```

This is your conda environment with TensorFlow & CUDA pre-installed.

To change it:
- `Ctrl + Shift + P` → "Python: Select Interpreter"
- Choose your preferred Python

---

## 🐛 Troubleshooting

### "CUDA not found"
```powershell
# Verify CUDA installation
nvidia-smi

# Check CUDA path
ls "C:\Program Files\NVIDIA\CUDA\v13\bin"
```

### GPU not detected in TensorFlow
1. Run `verify_cuda.py` to diagnose
2. Check `TF_CPP_MIN_LOG_LEVEL=0` in settings for detailed logs
3. Ensure CUDA 13 + cuDNN are installed

### "Python interpreter not found"
- Press `Ctrl + Shift + P`
- Type "Python: Select Interpreter"
- Choose the conda env path shown

### Terminal environment not updated
- Close VSCode completely
- Reopen VSCode
- Environment variables from settings.json will reload

---

## 📝 Next Steps

1. **Verify setup**: Run `verify_cuda.py`
2. **Run a test**: Use "Run Python with CUDA" task
3. **Debug code**: Set breakpoint → Press `F5`
4. **Monitor GPU**: Keep `nvidia-smi` running in separate terminal

---

## 📚 Additional Resources

- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
- TensorFlow GPU: https://www.tensorflow.org/install/gpu
- NVIDIA cuDNN: https://developer.nvidia.com/cudnn
- VSCode Python: https://code.visualstudio.com/docs/python/python-tutorial
