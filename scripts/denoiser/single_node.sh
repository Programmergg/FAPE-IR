#!/usr/bin/env bash
# set -euo pipefail

# ==== Parameters you can modify as needed ====
SESSION_NAME="FAPEIR"

LOG_DIR="xxxxx/logs"
LOG_FILE="$LOG_DIR/$SESSION_NAME.log"

# The Python you want to use (from the fapeir environment)
PY="xxxxx/envs/fapeir/bin/python"
ENV_BIN="xxxxx/envs/fapeir/bin"

# Main process port, can be overridden by external PORT environment variable
PORT=${PORT:-29501}

# ==== Log directory ====
mkdir -p "$LOG_DIR"

# ==== If a session with the same name already exists, clean it up first ====
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session '$SESSION_NAME' already exists. Killing the old session..."
  tmux kill-session -t "$SESSION_NAME"
fi

# ==== Create a new tmux session and start training ====
tmux new-session -d -s "$SESSION_NAME" -c "$(pwd)" bash -lc "
  set -euo pipefail
  exec &> >(tee -a '$LOG_FILE')

  echo '================= Starting training job ================='
  echo 'Session name: $SESSION_NAME'
  echo 'Log file: $LOG_FILE'
  echo 'Communication port: $PORT'

  # Safety: prepend fapeir/bin to PATH to avoid accidentally hitting executables from other environments
  export PATH='$ENV_BIN':\$PATH
  hash -r

  # Environment variables (adjust as needed)
  export TORCH_LOGS='recompiles'
  # export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export LD_PRELOAD=
  export TE_DISABLE_FP8=1
  # export NCCL_DEBUG=INFO
  export NCCL_ASYNC_ERROR_HANDLING=1
  export TORCH_NCCL_BLOCKING_WAIT=1
  export TORCHDYNAMO_DISABLE=1
  # export NCCL_NVLS_ENABLE=0
  # export NCCL_P2P_DISABLE=1
  export NCCL_IB_DISABLE=1
  # export NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL
  export NCCL_ALGO=Ring
  # export NCCL_PROTO=Simple
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  # NCCL / network optimization settings (adjust or remove as needed)
  export NCCL_IB_TC=136
  export NCCL_IB_SL=5
  export NCCL_IB_GID_INDEX=3
  export NCCL_SOCKET_IFNAME=eth
  export NCCL_IB_HCA=mlx5
  export NCCL_IB_TIMEOUT=22
  export NCCL_IB_QPS_PER_CONNECTION=8
  export NCCL_NET_PLUGIN=none

  # Multi-GPU settings
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  NUM_GPUS=8

  echo 'GPU count: ' \$NUM_GPUS
  echo 'CUDA devices: ' \$CUDA_VISIBLE_DEVICES

  echo '----------------- Runtime environment self-check -----------------'
  command -v accelerate >/dev/null 2>&1 && echo 'accelerate executable: ' \$(command -v accelerate) || echo 'accelerate executable: not found (no impact, this script directly uses python -m)'
  echo 'Python path: ' \$(\"$PY\" -c 'import sys; print(sys.executable)')

  # Steady-state self-check: do not fail or interrupt even if modules are missing
  \"$PY\" - <<'PYINFO' || true
import sys
print('PyVer:', sys.version.split()[0])

def torch_info():
    try:
        import torch
        cuda = getattr(getattr(torch, 'version', None), 'cuda', 'n/a')
        avail = torch.cuda.is_available() if hasattr(torch, 'cuda') else 'n/a'
        return f\"{getattr(torch,'__version__','n/a')} CUDA:{cuda} avail:{avail}\"
    except Exception as e:
        return f\"n/a ({type(e).__name__}: {e})\"

def ds_info():
    try:
        import deepspeed
        return f\"{getattr(deepspeed,'__version__','n/a')}\"
    except Exception as e:
        return f\"n/a ({type(e).__name__}: {e})\"

print('Torch:', torch_info())
print('DeepSpeed:', ds_info())
PYINFO
  echo '------------------------------------------------'

  # Start training (key point: directly invoke the accelerate module with fapeir's Python)
  \"$PY\" -m accelerate.commands.launch \
    --config_file scripts/accelerate_configs/single_node_zero2.yaml \
    --num_processes \${NUM_GPUS} \
    --main_process_port $PORT \
    flops.py \
    scripts/denoiser/flux_qwen2p5vl_7b_vlm_512.yaml

  echo 'Training job completed or interrupted'
  echo 'Press any key to exit...'
  read -n 1
"

echo "Training has been started in tmux session '$SESSION_NAME', log: $LOG_FILE"
echo "Attach to session: tmux attach -t $SESSION_NAME"
echo "Detach from session: Ctrl+B then press D"
echo "Kill session: tmux kill-session -t $SESSION_NAME"
echo "List sessions: tmux ls"