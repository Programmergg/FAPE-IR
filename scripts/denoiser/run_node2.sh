export MASTER_ADDR=172.31.16.20
export MASTER_PORT=29501
export NUM_MACHINES=3
export MACHINE_RANK=2
export GPUS_PER_NODE=8                 # 如用全卡，也可不设，脚本会自动探测
bash xxxxx/scripts/denoiser/multi_node.sh