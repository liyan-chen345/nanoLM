# Diffusion model training configuration
# Usage: python train.py config/train_diffusion.py

# Model type
model_type = 'diffusion'  # 'gpt2' or 'diffusion'

# I/O
out_dir = 'out-diffusion'
eval_interval = 500
log_interval = 10
sample_interval = 500  # Generate samples every N iterations
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch' or 'resume'

# Wandb logging
wandb_log = True
wandb_project = 'diffusion'
wandb_run_name = 'diffusion-shakespeare'

# Data
dataset = 'shakespeare'  # Will look for data/shakespeare/shakespeare.txt
batch_size = 64
block_size = 256  # Sequence length for diffusion

# Model architecture
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0
bias = False

# Diffusion-specific parameters
diffusion_steps = 128
context_len = 16  # Number of prefix tokens that are never masked
confidence_threshold = 0.95  # For confidence-based sampling

# Optimizer
learning_rate = 3e-4
max_iters = 20000
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 1000
lr_decay_iters = 20000
min_lr = 3e-5

# System
device = 'cuda'  # 'cpu', 'cuda', 'cuda:0', 'mps'
dtype = 'bfloat16'  # 'float32', 'bfloat16', 'float16'
compile = False  # Use PyTorch 2.0 compile (set to False for debugging)
