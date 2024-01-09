set -v

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=1
export CUBLAS_WORKSPACE_CONFIG=:16:8
export CUDA_VISIBLE_DEVICES=0
python distinctive_train_tuning.py