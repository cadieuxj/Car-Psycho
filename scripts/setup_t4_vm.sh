#!/bin/bash
# T4 VM Setup Script for Car-Psycho Platform
# This script installs all necessary components on a Ubuntu 22.04 VM with NVIDIA T4 GPU
# Run as: bash setup_t4_vm.sh

set -e  # Exit on error

echo "================================================"
echo "Car-Psycho T4 VM Setup Script"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Ubuntu
if [ ! -f /etc/lsb-release ]; then
    log_error "This script is designed for Ubuntu. Exiting."
    exit 1
fi

log_info "Starting T4 VM setup..."

# Update system
log_info "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install essential tools
log_info "Installing essential tools..."
sudo apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    htop \
    tmux \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install NVIDIA Driver
log_info "Installing NVIDIA Driver..."
if ! command -v nvidia-smi &> /dev/null; then
    log_info "NVIDIA driver not found. Installing..."
    sudo apt-get install -y ubuntu-drivers-common
    sudo ubuntu-drivers autoinstall
    log_warn "NVIDIA driver installed. System will need to reboot."
    log_warn "After reboot, run this script again to continue setup."
    read -p "Reboot now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo reboot
    fi
    exit 0
else
    log_info "NVIDIA driver already installed."
    nvidia-smi
fi

# Install CUDA Toolkit 12.1
log_info "Installing CUDA Toolkit..."
if ! command -v nvcc &> /dev/null; then
    log_info "CUDA not found. Installing CUDA 12.1..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit-12-1

    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc

    log_info "CUDA installed successfully."
else
    log_info "CUDA already installed."
    nvcc --version
fi

# Install Python 3.11
log_info "Installing Python 3.11..."
if ! command -v python3.11 &> /dev/null; then
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
    log_info "Python 3.11 installed."
else
    log_info "Python 3.11 already installed."
fi

python3 --version

# Install pip and upgrade
log_info "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install PyTorch with CUDA support
log_info "Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA
log_info "Verifying PyTorch CUDA installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Install Unsloth
log_info "Installing Unsloth for optimized training..."
pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"

# Install Axolotl
log_info "Installing Axolotl..."
git clone https://github.com/OpenAccess-AI-Collective/axolotl /opt/axolotl
cd /opt/axolotl
pip install packaging
pip install -e '.[flash-attn,deepspeed]'
cd ~

# Install additional ML libraries
log_info "Installing additional ML libraries..."
pip install \
    transformers>=4.38.0 \
    accelerate>=0.26.0 \
    peft>=0.8.0 \
    bitsandbytes>=0.42.0 \
    datasets>=2.16.0 \
    trl>=0.7.10 \
    scipy \
    scikit-learn \
    wandb \
    tensorboard

# Install Ollama
log_info "Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.ai/install.sh | sh
    log_info "Ollama installed successfully."
else
    log_info "Ollama already installed."
fi

# Start Ollama service
log_info "Starting Ollama service..."
sudo systemctl enable ollama
sudo systemctl start ollama

# Pull Llama 3.2 1B model
log_info "Pulling Llama 3.2 1B model..."
ollama pull llama3.2:1b

# Create workspace directories
log_info "Creating workspace directories..."
mkdir -p ~/carpsycho
mkdir -p ~/carpsycho/data
mkdir -p ~/carpsycho/models
mkdir -p ~/carpsycho/configs
mkdir -p ~/carpsycho/checkpoints
mkdir -p ~/carpsycho/logs

# Create training script directory
log_info "Setting up training environment..."
mkdir -p ~/carpsycho/scripts

# Install SSH server (if not installed)
log_info "Ensuring SSH server is installed..."
sudo apt-get install -y openssh-server
sudo systemctl enable ssh
sudo systemctl start ssh

# Configure firewall
log_info "Configuring firewall..."
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 11434/tcp   # Ollama
sudo ufw --force enable

# Install monitoring tools
log_info "Installing monitoring tools..."
pip install gpustat nvitop

# Create a systemd service for Ollama (if not exists)
log_info "Configuring Ollama systemd service..."
sudo tee /etc/systemd/system/ollama.service > /dev/null <<EOF
[Unit]
Description=Ollama Service
After=network.target

[Service]
Type=simple
User=$USER
Environment="OLLAMA_HOST=0.0.0.0:11434"
ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl restart ollama

# Create health check script
log_info "Creating health check script..."
cat > ~/carpsycho/scripts/health_check.sh <<'EOF'
#!/bin/bash
echo "=== System Health Check ==="
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
echo ""
echo "Ollama Status:"
curl -s http://localhost:11434/api/tags | python3 -m json.tool
echo ""
echo "Disk Usage:"
df -h | grep -E "Filesystem|/dev/sda"
echo ""
echo "Memory Usage:"
free -h
EOF

chmod +x ~/carpsycho/scripts/health_check.sh

# Create environment setup script
log_info "Creating environment setup script..."
cat > ~/carpsycho/scripts/setup_env.sh <<'EOF'
#!/bin/bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/axolotl:$PYTHONPATH
export HF_HOME=~/carpsycho/models/huggingface
export TRANSFORMERS_CACHE=~/carpsycho/models/transformers
EOF

chmod +x ~/carpsycho/scripts/setup_env.sh

# Add to bashrc
echo "source ~/carpsycho/scripts/setup_env.sh" >> ~/.bashrc

# Install requirements for remote execution
log_info "Installing Python packages for remote execution..."
pip install \
    paramiko \
    scp \
    pyyaml \
    python-dotenv

# Create a test training script
log_info "Creating test training script..."
cat > ~/carpsycho/scripts/test_training.py <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=== Training Environment Test ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print("\nLoading Llama 3.2 1B model...")
model_name = "meta-llama/Llama-3.2-1B"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print(f"Model loaded successfully!")
    print(f"Model parameters: {model.num_parameters() / 1e6:.2f}M")
except Exception as e:
    print(f"Error loading model: {e}")
    print("You may need to authenticate with HuggingFace: huggingface-cli login")

print("\n=== Test Complete ===")
EOF

# Final system info
log_info "Setup complete! System information:"
echo "=========================================="
python3 --version
nvcc --version
nvidia-smi
echo "=========================================="

log_info "Installation summary:"
log_info "✓ NVIDIA Driver installed"
log_info "✓ CUDA 12.1 installed"
log_info "✓ Python 3.11 installed"
log_info "✓ PyTorch with CUDA installed"
log_info "✓ Unsloth installed"
log_info "✓ Axolotl installed"
log_info "✓ Ollama installed and running"
log_info "✓ Llama 3.2 1B model pulled"
log_info "✓ Workspace created at ~/carpsycho"

echo ""
log_info "Next steps:"
echo "1. Authenticate with HuggingFace: huggingface-cli login"
echo "2. Test the environment: python3 ~/carpsycho/scripts/test_training.py"
echo "3. Check system health: bash ~/carpsycho/scripts/health_check.sh"
echo "4. Configure SSH key authentication for remote access"
echo ""
log_info "Ollama is running at: http://$(hostname -I | awk '{print $1}'):11434"
log_info "SSH is available at: ssh $USER@$(hostname -I | awk '{print $1}')"
echo ""

log_warn "IMPORTANT: Make sure to configure your .env file with:"
log_warn "  T4_VM_HOST=$(hostname -I | awk '{print $1}')"
log_warn "  T4_VM_USER=$USER"
echo ""

log_info "Setup complete! 🚀"
