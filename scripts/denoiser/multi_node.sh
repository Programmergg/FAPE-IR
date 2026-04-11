#!/usr/bin/env bash
set -euo pipefail

############################################
#   Can be modified as needed / overridden via env vars   #
############################################
SESSION_BASE_NAME="${SESSION_NAME:-FAPEIR}"
LOG_DIR="${LOG_DIR:-xxxxx/logs}"
PY="${PY:-xxxxx/conda_envs/fapeir/bin/python}"
ENV_BIN="${ENV_BIN:-xxxxx/conda_envs/fapeir/bin}"

# Accelerate config file (must include distributed_type: DEEPSPEED, etc.)
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-scripts/accelerate_configs/multi_node_zero2.yaml}"

############################################
#         Core multi-node multi-GPU params #
############################################
# Reachable IP or hostname of Rank0 machine (all machines should point to Rank0)
MASTER_ADDR="${MASTER_ADDR:?Please export MASTER_ADDR=rank0 node IP or hostname}"
MASTER_PORT="${MASTER_PORT:-29501}"

# Total number of machines (must be consistent across all machines)
NUM_MACHINES="${NUM_MACHINES:-2}"
# Rank of the current machine in [0, NUM_MACHINES-1]
MACHINE_RANK="${MACHINE_RANK:?Please export MACHINE_RANK=current machine rank (0..NUM_MACHINES-1)}"

# Number of GPUs used per machine; auto-detect if not set
if [[ -z "${GPUS_PER_NODE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS_PER_NODE=$(nvidia-smi -L | wc -l | tr -d ' ')
  else
    echo "nvidia-smi not found, defaulting GPUS_PER_NODE=8 (can be overridden via export)"
    GPUS_PER_NODE=8
  fi
fi

# (Optional) Manually specify visible GPUs
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

############################################
#            Other training params         #
############################################
PORT="${PORT:-$MASTER_PORT}"   # Compatible with your original variable name
SESSION_NAME="${SESSION_BASE_NAME}_rank${MACHINE_RANK}"
LOG_FILE="$LOG_DIR/$SESSION_NAME.log"

mkdir -p "$LOG_DIR"

############################################
#    If a session with the same name exists,
#      clean up the same rank first        #
############################################
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session '$SESSION_NAME' already exists, killing the old session..."
  tmux kill-session -t "$SESSION_NAME"
fi

############################################
#         Create tmux and launch training  #
############################################
tmux new-session -d -s "$SESSION_NAME" -c "$(pwd)" bash -lc "
  set -euo pipefail
  exec &> >(tee -a '$LOG_FILE')

  echo '================= Starting multi-node training ================='
  echo 'Session Name:         $SESSION_NAME'
  echo 'Log File:             $LOG_FILE'
  echo 'Master Address:       $MASTER_ADDR'
  echo 'Master Port:          $MASTER_PORT'
  echo 'NUM_MACHINES:         $NUM_MACHINES'
  echo 'MACHINE_RANK:         $MACHINE_RANK'
  echo 'GPUS_PER_NODE:        $GPUS_PER_NODE'
  echo 'ACCELERATE_CONFIG:    $ACCELERATE_CONFIG'

  # Ensure executables from the specified env are used
  export PATH=\"${ENV_BIN}:\$PATH\"
  hash -r

  # NCCL / network tuning (adjust as needed)
  export NCCL_IB_TC=\${NCCL_IB_TC:-136}
  export NCCL_IB_SL=\${NCCL_IB_SL:-5}
  export NCCL_IB_GID_INDEX=\${NCCL_IB_GID_INDEX:-3}
  export NCCL_SOCKET_IFNAME=\${NCCL_SOCKET_IFNAME:-eth}      # If using IB, can be set to mlx5_0 or bond0, etc.
  export NCCL_IB_HCA=\${NCCL_IB_HCA:-mlx5}
  export NCCL_IB_TIMEOUT=\${NCCL_IB_TIMEOUT:-22}
  export NCCL_IB_QPS_PER_CONNECTION=\${NCCL_IB_QPS_PER_CONNECTION:-8}
  export NCCL_NET_PLUGIN=\${NCCL_NET_PLUGIN:-none}
  export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'
  export TORCH_LOGS='recompiles'
  # Enable for debugging if needed: export NCCL_DEBUG=INFO

  echo '----------------- Runtime self-check -----------------'
  command -v accelerate >/dev/null 2>&1 && echo 'accelerate executable: ' \$(command -v accelerate) || echo 'accelerate executable: not found (does not matter, this script uses python -m)'
  echo 'Python path: ' \$(\"$PY\" -c 'import sys; print(sys.executable)')

  \"$PY\" - <<'PYINFO' || true
import os, sys
print('PyVer:', sys.version.split()[0])
def torch_info():
    try:
        import torch
        cuda = getattr(getattr(torch, 'version', None), 'cuda', 'n/a')
        avail = torch.cuda.is_available() if hasattr(torch, 'cuda') else 'n/a'
        return f\"{getattr(torch,'__version__','n/a')} CUDA:{cuda} avail:{avail} devs:{torch.cuda.device_count() if hasattr(torch,'cuda') else 'n/a'}\"
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
print('ENV CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('LOCAL_RANK:', os.environ.get('LOCAL_RANK'), 'RANK:', os.environ.get('RANK'), 'WORLD_SIZE:', os.environ.get('WORLD_SIZE'))
PYINFO
  echo '------------------------------------------------------'

  # Compute total number of processes and GPU list for this node
  TOTAL_PROCESSES=$(( NUM_MACHINES * GPUS_PER_NODE ))
  GPU_IDS=\$(seq -s, 0 \$((GPUS_PER_NODE-1)))
  echo 'TOTAL_PROCESSES:      ' \$TOTAL_PROCESSES
  echo 'GPU_IDS:              ' \$GPU_IDS

  echo '================= Launching accelerate ==============='
  \"$PY\" -m accelerate.commands.launch \\
    --config_file \"$ACCELERATE_CONFIG\" \\
    --num_machines      \"\${NUM_MACHINES}\" \\
    --machine_rank      \"\${MACHINE_RANK}\" \\
    --num_processes     \"\$TOTAL_PROCESSES\" \\
    --main_process_ip   \"$MASTER_ADDR\" \\
    --main_process_port \"$MASTER_PORT\" \\
    --gpu_ids           \"\$GPU_IDS\" \\
    train_and_test.py \\
    scripts/denoiser/flux_qwen2p5vl_7b_vlm_512.yaml

  echo 'Training job completed or interrupted'
  echo 'Press any key to exit...'
  read -n 1
"

echo "Training has been started in tmux session '$SESSION_NAME', log: $LOG_FILE"
echo "Attach to session: tmux attach -t $SESSION_NAME"
echo "Detach from session: Ctrl+B then D"
echo "Kill session: tmux kill-session -t $SESSION_NAME"
echo "List sessions: tmux ls"