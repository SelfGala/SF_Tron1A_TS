export CUDA_VISIBLE_DEVICES=1
python scripts/rsl_rl/train.py \
    --task Isaac-SF-Blind-Flat-v1 \
    --max_iterations 15000 \
    --headless