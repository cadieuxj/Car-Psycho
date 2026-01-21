# Car-Psycho: Psychometric Car Sales Platform

> **The Artificial Psychologist for Car Sales**
> A production-ready platform implementing cutting-edge psychometric profiling research adapted for automotive sales.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-orange.svg)
![FastAPI](https://img.shields.io/badge/fastapi-0.109+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Overview

Car-Psycho is a comprehensive platform that applies the **"Architecting the Artificial Psychologist"** research methodology to the automotive sales domain. It uses a fine-tuned **Llama 3.2 1B** model to perform real-time psychometric profiling (Big Five/OCEAN traits) from customer car preferences and conversations.

### Key Innovations

1. **PsychSteer**: Novel synthetic labeling technique that maps car preferences → personality traits
2. **RegMix**: Regression-based data mixing (40% public psychology data, 40% car domain data, 20% synthetic)
3. **Ordinal Regression**: Custom loss function respecting the ordinal nature of personality traits
4. **Mixture of Losses (MoL)**: Cross-Entropy for domain data + KL Divergence for general data
5. **PostToPersonality (P2P)**: RAG-based inference framework for real-time profiling

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (Next.js)                      │
│  ┌──────────────────┐        ┌─────────────────────────┐  │
│  │  ModelControl    │        │   SalesAssistant        │  │
│  │  - Train Models  │        │   - Chat Interface      │  │
│  │  - Upload Data   │        │   - OCEAN Dashboard     │  │
│  │  - View Progress │        │   - Recommendations     │  │
│  └──────────────────┘        └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Nginx Gateway   │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐   ┌────────▼────────┐   ┌───────▼───────┐
│ Manager Service│   │Inference Service│   │  Data Service │
│ - Train Jobs   │   │ - P2P Framework │   │ - PsychSteer  │
│ - Model Reg    │   │ - RAG Engine    │   │ - RegMix      │
│ - Remote Exec  │   │ - OCEAN Scoring │   │ - Downloaders │
└────────┬───────┘   └────────┬────────┘   └───────┬───────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
    ┌────▼────┐       ┌───────▼──────┐      ┌─────▼─────┐
    │PostgreSQL│       │   ChromaDB   │      │   Redis   │
    │  Relational     │   Vector DB   │      │  Cache    │
    └─────────┘       └──────────────┘      └───────────┘

                              │
                    ┌─────────▼─────────┐
                    │   T4 GPU VM       │
                    │ - Ollama          │
                    │ - Unsloth         │
                    │ - Axolotl         │
                    │ - Training Jobs   │
                    └───────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Local Machine**: Docker, Docker Compose, 8GB+ RAM
- **Remote T4 VM**: Ubuntu 22.04, NVIDIA T4 GPU, 16GB+ RAM, CUDA 12.1
- **API Keys**: OpenAI or Anthropic (for PsychSteer teacher model)

### 1. Clone and Configure

```bash
git clone <repository-url>
cd Car-Psycho

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env
```

**Required .env variables:**
```bash
# API Keys (get from OpenAI or Anthropic)
OPENAI_API_KEY=sk-your-key-here
# OR
ANTHROPIC_API_KEY=sk-ant-your-key-here

# T4 VM Configuration
T4_VM_HOST=your-vm-ip-address
T4_VM_USER=ubuntu
T4_VM_SSH_KEY_PATH=~/.ssh/id_rsa
```

### 2. Set Up T4 VM

SSH into your T4 VM and run the setup script:

```bash
# Copy setup script to VM
scp scripts/setup_t4_vm.sh user@your-vm-ip:~/

# SSH into VM
ssh user@your-vm-ip

# Run setup (this will take 20-30 minutes)
bash setup_t4_vm.sh

# After first reboot (if needed), run again to complete setup
bash setup_t4_vm.sh

# Verify installation
bash ~/carpsycho/scripts/health_check.sh
```

### 3. Start Local Services

```bash
# Start all services
docker-compose up -d

# Check service health
curl http://localhost:8080/health
curl http://localhost:8080/api/health/manager
curl http://localhost:8080/api/health/inference
curl http://localhost:8080/api/health/data

# View logs
docker-compose logs -f
```

### 4. Run Data Pipeline

```bash
# Enter the data service container
docker exec -it carpsycho-data bash

# Run the complete data ingestion pipeline
cd /app/backend/services/data/core
python data_ingest.py --teacher-model gpt-4o --total-samples 1000

# This will:
# 1. Download Big5-Chat and PANDORA datasets
# 2. Load car questionnaires from data/car_questionnaire/
# 3. Apply PsychSteer labeling using GPT-4o
# 4. Apply RegMix (40:40:20) data mixing
# 5. Output training data to data/processed/
```

### 5. Access Frontend

```bash
# Frontend will be available at:
http://localhost:3000

# API Gateway at:
http://localhost:8080
```

## 📊 Data Pipeline (Phase 1)

### PsychSteer: Synthetic Labeling

The PsychSteer component is located in `backend/services/data/core/psychsteer.py`.

**Example usage:**

```python
from psychsteer import PsychSteer, CarProfileInput, TeacherModel

# Initialize with teacher model
psychsteer = PsychSteer(
    teacher_model=TeacherModel.GPT4O,
    temperature=0.3
)

# Create a car profile
profile = CarProfileInput(
    customer_id="CUST_001",
    responses={
        "Budget Range": "$35,000 - $45,000",
        "Primary Use": "Daily commute and family trips",
        "Key Priorities": "Safety ratings, reliability, low maintenance",
        # ... more questions
    },
    raw_text="I want a safe car for my family..."
)

# Label with Big Five traits
result = await psychsteer.label_car_profile(profile)

# Result contains:
# {
#   "personality_scores": {
#     "openness": 0.65,
#     "conscientiousness": 0.88,
#     "extraversion": 0.55,
#     "agreeableness": 0.75,
#     "neuroticism": 0.42
#   },
#   "confidence_scores": {...},
#   "reasoning": {...}
# }
```

**Research-Based Mappings:**

| Car Preference | Big Five Trait | Score Range |
|---------------|---------------|-------------|
| Innovative features, electric vehicles | **High Openness** | 0.7-1.0 |
| Safety ratings, warranties, detailed research | **High Conscientiousness** | 0.7-1.0 |
| Sporty, attention-grabbing, performance | **High Extraversion** | 0.7-1.0 |
| Family-oriented, comfort, affordability | **High Agreeableness** | 0.7-1.0 |
| Excessive safety concerns, indecisiveness | **High Neuroticism** | 0.7-1.0 |

### RegMix: Data Mixing Strategy

The RegMix component is in `backend/services/data/core/regmix.py`.

**Configuration:**

```python
from regmix import RegMix, RegMixConfig, DataSource

# Configure mixing ratios
config = RegMixConfig(
    public_general_ratio=40,  # General psychology data
    car_domain_ratio=40,      # Car-specific data
    synthetic_ratio=20,       # PsychSteer labeled data
    total_samples=10000,
    stratify_by_traits=True   # Maintain trait distribution
)

# Initialize and mix
regmix = RegMix(config)
mixed_data, stats = regmix.mix_datasets(
    public_sources=[big5_source, pandora_source],
    car_sources=[car_questionnaire_source],
    synthetic_sources=[psychsteer_source]
)

# Output statistics show trait distributions
print(stats["trait_distributions"])
```

## 🧠 Training Engine (Phase 2)

### OrdinalPsychTrainer: Custom Loss Functions

The trainer is located in `backend/ml/trainers/ordinal_psych_trainer.py`.

**Key Features:**

1. **Ordinal Regression Loss** (`backend/ml/losses/ordinal_regression.py`):
   - Respects ordinal nature of personality scores
   - Penalizes distant errors more heavily
   - Example: Predicting 0.9 when true is 0.2 (error 0.7) is penalized MORE than predicting 0.5 when true is 0.2 (error 0.3)

2. **Mixture of Losses** (`backend/ml/losses/mixture_of_losses.py`):
   - Cross-Entropy for car domain data (learn specific patterns)
   - KL Divergence for general psychology data (preserve knowledge)
   - Combined loss for synthetic data

**Usage Example:**

```python
from trainers.ordinal_psych_trainer import (
    OrdinalPsychTrainer,
    OrdinalPsychTrainingArguments
)

# Configure training
args = OrdinalPsychTrainingArguments(
    output_dir="./models/checkpoints",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,

    # Custom loss weights
    lm_loss_weight=0.6,              # Language modeling
    personality_loss_weight=0.4,      # Personality prediction
    ordinal_loss_weight=0.7,          # Ordinal regression
    mse_loss_weight=0.3,              # MSE component

    # MoL configuration
    ce_weight=1.0,                    # Cross-entropy weight
    kl_weight=0.5,                    # KL divergence weight

    # Domain boost
    car_domain_boost=1.2,             # 20% boost for car data

    # Personality prediction
    predict_personality=True,
    personality_head_hidden_size=256
)

# Initialize trainer
trainer = OrdinalPsychTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# Train
trainer.train()

# Save loss history for visualization
trainer.save_loss_history("./models/checkpoints/loss_history.json")
```

### Training Orchestration

The training orchestrator manages remote training jobs on the T4 VM.

**Workflow:**

1. Generate Axolotl configuration
2. Upload training data and config to T4 VM
3. Execute training via SSH
4. Monitor progress via logs
5. Download trained model

**Configuration Example (Axolotl):**

```yaml
# Auto-generated config for Llama 3.2 1B
base_model: meta-llama/Llama-3.2-1B
model_type: LlamaForCausalLM

# Dataset
datasets:
  - path: /home/ubuntu/carpsycho/data/training_data.jsonl
    type: completion

# Training
adapter: lora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05

sequence_len: 2048
batch_size: 4
gradient_accumulation_steps: 4
num_epochs: 3
learning_rate: 2e-4

# Optimization (Unsloth)
bf16: true
tf32: true
gradient_checkpointing: true
flash_attention: true

# Ordinal Psychology (custom)
custom_trainer: ordinal_psych_trainer.OrdinalPsychTrainer
trainer_args:
  lm_loss_weight: 0.6
  personality_loss_weight: 0.4
  ordinal_loss_weight: 0.7
  mse_loss_weight: 0.3
```

## 🔮 Inference Engine (Phase 3)

### PostToPersonality (P2P) Framework

The P2P inference engine combines RAG with personality modeling.

**Flow:**

```
User Query → Retrieve Similar Profiles → Generate with Context → Parse OCEAN Scores
     ↓              (ChromaDB)                  (Llama 3.2)              ↓
 Embedding ────────────────────────────────────────────────────→ JSON Output
```

**API Endpoint:**

```bash
POST /api/analyze/profile
Content-Type: application/json

{
  "session_id": "sess_123",
  "questionnaire": {
    "budget": "$30,000-$40,000",
    "priorities": "Safety and reliability",
    "family_size": 4
  },
  "conversation_history": [
    {"role": "customer", "content": "I need a safe car for my family..."}
  ]
}

# Response:
{
  "personality_scores": {
    "openness": 0.62,
    "conscientiousness": 0.87,
    "extraversion": 0.54,
    "agreeableness": 0.76,
    "neuroticism": 0.41
  },
  "confidence": {
    "openness": 0.82,
    "conscientiousness": 0.91,
    ...
  },
  "car_recommendations": [
    {
      "model": "Honda CR-V",
      "reasoning": "High safety ratings align with conscientiousness...",
      "match_score": 0.94
    }
  ],
  "sales_strategy": {
    "approach": "Emphasize safety features, reliability data, and warranty...",
    "tone": "Professional, detail-oriented, reassuring"
  }
}
```

## 📱 Frontend (Phase 4)

### ModelControl Component

Location: `frontend/src/components/ModelControl/`

**Features:**
- Upload training datasets
- Configure RegMix ratios (visual sliders)
- Start training jobs on T4 VM
- Real-time training progress (loss curves, epoch counter)
- Model registry and versioning

### SalesAssistant Component

Location: `frontend/src/components/SalesAssistant/`

**Features:**

```
┌─────────────────────────────────────────────────────────┐
│                   Sales Assistant                        │
├──────────────────────────┬──────────────────────────────┤
│   Chat Panel (Left)      │  Psychometric Dashboard      │
│                          │  (Right)                     │
│  Customer: "I need a     │  ┌────────────────────────┐ │
│  safe car..."            │  │  OCEAN Scores          │ │
│                          │  │  O: ████████░░ 0.82    │ │
│  Agent: "Based on your   │  │  C: █████████░ 0.91    │ │
│  priorities..."          │  │  E: █████░░░░░ 0.54    │ │
│                          │  │  A: ████████░░ 0.76    │ │
│  [Thought Trace]         │  │  N: ████░░░░░░ 0.41    │ │
│  "High conscientiousness │  └────────────────────────┘ │
│  detected. Emphasize     │                              │
│  safety and reliability" │  Recommended: Honda CR-V     │
│                          │  Match: 94%                  │
└──────────────────────────┴──────────────────────────────┘
```

## 🗂️ Project Structure

```
Car-Psycho/
├── backend/
│   ├── services/
│   │   ├── manager/         # Training orchestration
│   │   ├── inference/       # P2P inference engine
│   │   └── data/           # Data pipeline
│   ├── ml/
│   │   ├── trainers/       # OrdinalPsychTrainer
│   │   ├── losses/         # Custom losses
│   │   └── utils/
│   └── shared/             # Shared utilities
│
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── ModelControl/
│       │   └── SalesAssistant/
│       └── lib/
│
├── data/
│   ├── raw/               # Downloaded datasets
│   ├── processed/         # RegMix output
│   ├── car_questionnaire/ # Proprietary data
│   └── synthetic/         # PsychSteer labeled
│
├── models/
│   ├── checkpoints/       # Training checkpoints
│   ├── fine_tuned/       # Final models
│   └── configs/          # Axolotl configs
│
├── scripts/
│   ├── setup_t4_vm.sh    # T4 VM setup
│   └── init_db.sql       # Database schema
│
├── docker/
│   ├── backend.Dockerfile
│   ├── frontend.Dockerfile
│   └── nginx.conf
│
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 📖 Research Implementation

### Methodology Source

This platform implements the methodology from:
**"Architecting the Artificial Psychologist: Integrating Personality Profiling into AI Systems"**

### Implementation Mapping

| Research Component | Implementation | Location |
|-------------------|----------------|----------|
| **PostToPersonality (P2P)** | RAG-based inference | `backend/services/inference/core/inference_engine.py` |
| **PsychSteer** | Synthetic labeling | `backend/services/data/core/psychsteer.py` |
| **RegMix** | Data mixing (40:40:20) | `backend/services/data/core/regmix.py` |
| **Ordinal Regression** | Custom loss | `backend/ml/losses/ordinal_regression.py` |
| **MoL** | Mixed losses | `backend/ml/losses/mixture_of_losses.py` |
| **OrdinalPsychTrainer** | Custom trainer | `backend/ml/trainers/ordinal_psych_trainer.py` |
| **Unsloth** | Optimization | T4 VM setup script |
| **DPO** | Alignment | Axolotl config |

## 🔧 Advanced Configuration

### Custom Questionnaire Schema

Add your own questionnaire questions in `data/car_questionnaire/`:

```json
{
  "customer_id": "CUST_001",
  "responses": {
    "Your Question Here": "Customer answer",
    "Another Question": "Another answer"
  },
  "raw_text": "Optional free-form text"
}
```

### Model Selection

Switch between models via environment variables:

```bash
# Use fine-tuned model
ACTIVE_MODEL=car-psych-v1

# Use generic Llama 3.2
ACTIVE_MODEL=llama3.2:1b

# Use external API
ACTIVE_MODEL=gpt-4o
```

### Training Hyperparameters

Edit `models/configs/axolotl_config.yaml`:

```yaml
learning_rate: 2e-4      # Adjust learning rate
lora_r: 16               # LoRA rank
batch_size: 4            # Batch size
num_epochs: 3            # Training epochs
```

## 📊 Monitoring & Debugging

### Check Training Progress

```bash
# View training logs on T4 VM
ssh user@t4-vm
tail -f ~/carpsycho/logs/training.log

# View loss history
cat ~/carpsycho/models/checkpoints/loss_history.json
```

### Database Queries

```bash
# Connect to PostgreSQL
docker exec -it carpsycho-postgres psql -U carpsycho

# Check training jobs
SELECT job_name, status, progress, current_epoch
FROM training_jobs
WHERE status = 'running';

# Check models
SELECT name, version, status, is_default
FROM models;
```

### ChromaDB Status

```bash
# Check vector collections
curl http://localhost:8000/api/v1/collections
```

## 🚨 Troubleshooting

### Common Issues

**Issue**: Training job fails to start on T4 VM
```bash
# Check VM connectivity
ssh user@t4-vm "nvidia-smi"

# Check Ollama status
ssh user@t4-vm "systemctl status ollama"

# Verify Unsloth installation
ssh user@t4-vm "python3 -c 'import unsloth; print(unsloth.__version__)'"
```

**Issue**: PsychSteer labeling fails
```bash
# Verify API keys
echo $OPENAI_API_KEY

# Test API connection
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

**Issue**: Docker services won't start
```bash
# Check disk space
df -h

# Clean Docker
docker system prune -a

# Rebuild
docker-compose down
docker-compose up --build
```

## 📚 API Documentation

### Complete API Reference

After starting services, access:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/training/start` | POST | Start training job |
| `/api/training/status/{job_id}` | GET | Check training status |
| `/api/models` | GET | List available models |
| `/api/analyze/profile` | POST | Analyze customer profile |
| `/api/datasets/upload` | POST | Upload training data |
| `/api/psychsteer/label` | POST | Label car questionnaire |

## 🤝 Contributing

This is a research implementation. For production use, consider:

1. Security hardening (API authentication, rate limiting)
2. Scalability improvements (distributed training, caching)
3. Model versioning (MLflow, DVC)
4. Comprehensive testing
5. GDPR compliance for personality data

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Research methodology: "Architecting the Artificial Psychologist"
- Base model: Llama 3.2 by Meta
- Optimization: Unsloth
- Training framework: Axolotl
- Big Five personality model: academic psychology research

## 📧 Support

For issues, questions, or contributions:
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: [your-email]

---

**Built with 🧠 for the future of personalized sales**

*Making car sales less about persuasion, more about understanding*
