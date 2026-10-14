#!/usr/bin/env bash
# ONNX Runtime + CUDA 环境（30HX 推理用）
# 用法： source tools/ort_env.sh
#
# 背景：pip 装的 CUDA 运行库在 venv 的 site-packages/nvidia/*/lib 下，
# 不在系统库搜索路径里，ONNX Runtime 加载 CUDAExecutionProvider 时会报
# "libcublasLt.so.12: cannot open shared object file"。
# 必须把这些目录加进 LD_LIBRARY_PATH 才能用上 GPU。
SP="${SP:-$HOME/mjlab-env/venv/lib/python3.10/site-packages}"
ORT_LIBS="$SP/nvidia/cublas/lib:$SP/nvidia/cudnn/lib:$SP/nvidia/cuda_runtime/lib:$SP/nvidia/cuda_nvrtc/lib:$SP/nvidia/cufft/lib:$SP/nvidia/curand/lib:$SP/nvidia/cusolver/lib:$SP/nvidia/cusparse/lib:$SP/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$ORT_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ORT_CUDA=1
echo "ORT CUDA 环境已就绪："
echo "  venv      = $HOME/mjlab-env/venv"
echo "  LD_LIBRARY_PATH 已追加 $(echo $ORT_LIBS | tr ':' '\n' | wc -l) 个 nvidia 库目录"
