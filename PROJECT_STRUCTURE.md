# Car-Psycho: Psychometric Car Sales Platform
## Project Structure

```
Car-Psycho/
├── backend/
│   ├── services/
│   │   ├── manager/                    # Manager Service
│   │   │   ├── __init__.py
│   │   │   ├── main.py                # FastAPI app
│   │   │   ├── api/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── training.py        # Training job endpoints
│   │   │   │   ├── models.py          # Model registry endpoints
│   │   │   │   └── config.py          # Configuration endpoints
│   │   │   ├── core/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── train_orchestrator.py  # Core training orchestration
│   │   │   │   ├── remote_executor.py     # SSH/Remote execution
│   │   │   │   └── config_generator.py    # Axolotl config generation
│   │   │   ├── models/
│   │   │   │   ├── __init__.py
│   │   │   │   └── schemas.py         # Pydantic schemas
│   │   │   └── requirements.txt
│   │   │
│   │   ├── inference/                  # Inference Service
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── api/
│   │   │   │   ├── __init__.py
│   │   │   │   └── inference.py       # Inference endpoints
│   │   │   ├── core/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── inference_engine.py    # P2P implementation
│   │   │   │   ├── rag_engine.py          # ChromaDB RAG
│   │   │   │   └── model_adapter.py       # Ollama/API adapters
│   │   │   ├── models/
│   │   │   │   ├── __init__.py
│   │   │   │   └── schemas.py
│   │   │   └── requirements.txt
│   │   │
│   │   └── data/                       # Data Service
│   │       ├── __init__.py
│   │       ├── main.py
│   │       ├── api/
│   │       │   ├── __init__.py
│   │       │   └── datasets.py        # Dataset management endpoints
│   │       ├── core/
│   │       │   ├── __init__.py
│   │       │   ├── data_ingest.py     # Main data ingestion
│   │       │   ├── psychsteer.py      # PsychSteer implementation
│   │       │   ├── regmix.py          # RegMix implementation
│   │       │   └── downloaders.py     # Dataset downloaders
│   │       ├── models/
│   │       │   ├── __init__.py
│   │       │   └── schemas.py
│   │       └── requirements.txt
│   │
│   ├── shared/
│   │   ├── __init__.py
│   │   ├── database.py                # PostgreSQL connection
│   │   ├── chromadb_client.py         # ChromaDB connection
│   │   ├── config.py                  # Shared configuration
│   │   └── utils.py                   # Shared utilities
│   │
│   └── ml/
│       ├── __init__.py
│       ├── trainers/
│       │   ├── __init__.py
│       │   └── ordinal_psych_trainer.py   # Custom trainer with MoL
│       ├── losses/
│       │   ├── __init__.py
│       │   ├── ordinal_regression.py      # Ordinal regression loss
│       │   └── mixture_of_losses.py       # MoL implementation
│       └── utils/
│           ├── __init__.py
│           └── model_utils.py
│
├── frontend/
│   ├── package.json
│   ├── next.config.js
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx
│   │   │   ├── dashboard/
│   │   │   │   └── page.tsx           # Training dashboard
│   │   │   └── assistant/
│   │   │       └── page.tsx           # Sales assistant interface
│   │   ├── components/
│   │   │   ├── ModelControl/
│   │   │   │   ├── index.tsx          # Model training UI
│   │   │   │   ├── TrainingProgress.tsx
│   │   │   │   ├── DatasetUpload.tsx
│   │   │   │   └── RegMixConfig.tsx
│   │   │   ├── SalesAssistant/
│   │   │   │   ├── index.tsx          # Main assistant interface
│   │   │   │   ├── ChatPanel.tsx      # Left: Chat
│   │   │   │   ├── PsychometricDashboard.tsx  # Right: OCEAN scores
│   │   │   │   ├── ThoughtTrace.tsx   # Chain of thought display
│   │   │   │   └── CarRecommendation.tsx
│   │   │   └── shared/
│   │   │       ├── Sidebar.tsx
│   │   │       └── Header.tsx
│   │   ├── lib/
│   │   │   ├── api.ts                 # API client
│   │   │   └── types.ts               # TypeScript types
│   │   └── hooks/
│   │       ├── useWebSocket.ts        # Real-time updates
│   │       └── useTraining.ts
│   │
│   └── public/
│       └── assets/
│
├── scripts/
│   ├── setup_t4_vm.sh                 # T4 VM setup script
│   ├── deploy_backend.sh
│   └── init_db.sql                    # Database initialization
│
├── data/
│   ├── raw/                           # Downloaded datasets
│   ├── processed/                     # Processed datasets
│   ├── car_questionnaire/             # Proprietary data
│   └── synthetic/                     # PsychSteer generated
│
├── models/
│   ├── checkpoints/                   # Training checkpoints
│   ├── fine_tuned/                    # Final models
│   └── configs/                       # Training configs
│
├── docker/
│   ├── backend.Dockerfile
│   ├── frontend.Dockerfile
│   └── nginx.conf
│
├── docker-compose.yml
├── .env.example
├── README.md
└── requirements.txt                   # Root requirements
```

## Architecture Overview

### Research Implementation Map

| Research Component | Implementation Location | Description |
|-------------------|------------------------|-------------|
| **P2P Framework** | `backend/services/inference/core/inference_engine.py` | PostToPersonality RAG-based inference |
| **PsychSteer** | `backend/services/data/core/psychsteer.py` | Synthetic labeling with teacher LLM |
| **RegMix** | `backend/services/data/core/regmix.py` | Regression-based data mixing (40:40:20) |
| **Ordinal Regression** | `backend/ml/losses/ordinal_regression.py` | Custom loss for personality traits |
| **MoL (Mixture of Losses)** | `backend/ml/losses/mixture_of_losses.py` | Cross-Entropy + KL Divergence |
| **OrdinalPsychTrainer** | `backend/ml/trainers/ordinal_psych_trainer.py` | Custom HuggingFace Trainer |
| **Unsloth Integration** | `backend/services/manager/core/train_orchestrator.py` | Remote training execution |
| **DPO Alignment** | `models/configs/axolotl_config.yaml` | Direct Preference Optimization |

### Data Flow

1. **Training Pipeline**:
   ```
   Raw Data → PsychSteer Labeling → RegMix → Training → Fine-tuned Model
   ```

2. **Inference Pipeline**:
   ```
   User Query → RAG Retrieval → P2P Context → Llama 3.2 → OCEAN Scores
   ```

### Tech Stack

- **Backend**: FastAPI, PostgreSQL, ChromaDB
- **ML**: PyTorch, Transformers, Unsloth, Axolotl, Ollama
- **Frontend**: Next.js 14, React, TailwindCSS, shadcn/ui
- **Infrastructure**: Docker, SSH/Paramiko, T4 GPU VM
- **APIs**: OpenAI/Anthropic for PsychSteer teacher model

## Key Innovation Points

1. **Domain Adaptation**: Car sales questionnaire → Big Five traits via PsychSteer
2. **Ordinal Psychology**: Custom loss function respecting trait ordinality
3. **Hybrid Compute**: Local orchestration + remote GPU training
4. **Real-time Profiling**: Live OCEAN score updates during conversation
