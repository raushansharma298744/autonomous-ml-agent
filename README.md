# AutoML Agent – Autonomous Machine Learning System

An agentic AI system that autonomously performs end-to-end machine learning workflows on user-provided datasets.

## Overview

This project implements a truly autonomous ML engineer using LangGraph for orchestration. Unlike traditional AutoML tools that simply run a fixed pipeline, this system demonstrates genuine agentic behavior:

**OBSERVE → ANALYZE → PLAN → ACT → EXECUTE → EVALUATE → CRITIQUE → IMPROVE**

The agent inspects ML results, identifies weaknesses, decides on improvements, executes them, evaluates again, and continues iterating until a stopping condition is reached.

## Features

- **Autonomous ML Workflow**: End-to-end from dataset to production-ready model
- **Agentic Architecture**: Multiple specialized agents orchestrated via LangGraph
- **Real Feedback Loop**: Critic agent analyzes results and recommends improvements
- **Experiment Tracking**: Full MLflow integration with PostgreSQL metadata
- **Modern Stack**: FastAPI backend, React frontend, PostgreSQL, MLflow
- **Production Ready**: Type hints, validation, error handling, logging, Docker support

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LangGraph Orchestrator                    │
├─────────────────────────────────────────────────────────────────┤
│  Dataset Analyst → Planner → Preprocessing → EDA → Features     │
│       ↓                                                          │
│  Model Selection → Training → Evaluation → Critic               │
│       ↓                      ↑              ↓                    │
│  Hyperparameter Tuning ──────┘              │                    │
│       ↓                                     │                    │
│  Improvement Loop ◄─────────────────────────┘                    │
│       ↓                                                          │
│  Final Report                                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Autonomous Agent Architecture (Implemented)

The compiled LangGraph workflow implements a genuine autonomous feedback loop:

```mermaid
flowchart TD
    START --> DA[Dataset Analysis]
    DA --> PL[Planner]
    PL --> PP[Preprocessing]
    PP --> EDA[EDA]
    EDA --> FE[Feature Engineering]
    FE --> MS[Model Selection]
    MS --> TU[Tuning]
    TU --> CR[Critic]
    CR -->|continue| IR[Improvement Router]
    CR -->|stop| FR[Generate Final Report]
    IR -->|class_weight| CW[Apply Class Weight]
    IR -->|resample| RS[Apply Resampling]
    IR -->|smote| SM[Apply SMOTE]
    IR -->|threshold_adjustment| TH[Adjust Threshold]
    IR -->|feature_engineering| FE2[Engineer Features]
    IR -->|different_model| DM[Select Different Model]
    IR -->|hyperparameter_tuning| HT[Run Hyperparameter Tuning]
    CW --> MS
    RS --> MS
    SM --> MS
    TH --> MS
    FE2 --> MS
    DM --> MS
    HT --> MS
    FR --> END
```

#### State (`AgentState`)
Key fields persisted across the loop:
- `dataset_profile`, `problem_type`, `target_column`
- `workflow_plan`, `planner_summary`
- `preprocessing_summary`, `preprocessing_actions`
- `eda_results`, `eda_summary`
- `feature_engineering_summary`, `feature_actions`
- `experiments[]`, `best_experiment`, `best_metrics`
- `critic_feedback`, `should_continue`, `stopping_reason`
- `iteration_number`, `max_iterations`, `improvement`
- `improvement_history[]`, `next_action`
- `final_report`

#### Agents (LangGraph Nodes)
| Node | Responsibility |
|------|----------------|
| `dataset_analysis` | Load CSV, profile, detect problem type, imbalance, ID columns |
| `planner` | LLM‑driven workflow plan + candidate model list |
| `preprocessing` | Train‑test split, impute, encode, scale, class‑weight calc |
| `eda` | Target distribution, correlations, missingness, outliers → structured insights |
| `feature_engineering` | Log‑transform skew, datetime decompose, interactions, drop redundant |
| `model_selection` | Train baseline models (LR, RF, GB, XGB), evaluate, pick best |
| `tuning` | Optuna (TPE) on promising model, CV scoring, retrain best |
| `critic` | Rule‑based + optional LLM analysis → issues + recommended actions |
| `improvement_router` | Maps critic primary action → concrete improvement node |
| `apply_class_weight` … `run_hyperparameter_tuning` | Lightweight nodes that mutate state (e.g., set `class_weights="balanced"`) then loop to `model_selection` |
| `generate_final_report` | Assemble JSON + HTML report from all accumulated state |

#### Conditional Edges
- `critic` → `should_continue_improvement` → `"continue"` → `improvement_router` / `"stop"` → `generate_final_report`
- `improvement_router` → `select_next_action` → one of 7 improvement nodes or `generate_final_report`

#### Stopping Conditions
1. `iteration_number >= max_iterations` (default 5)
2. `best_metrics[f1|r2] >= 0.95|0.90`
3. `improvement < MIN_IMPROVEMENT_THRESHOLD` (default 0.005)
4. Critic status `"satisfactory"`
5. No further improvement actions available

#### LLM Integration
- Centralised `LLMService` (`app/services/llm_service.py`) using Groq `ChatGroq`.
- Used by **Planner** (workflow plan), **Critic** (extra reasoning), **Feature Engineer** (optional), **Report** (narrative).
- Graceful deterministic fallbacks when API key missing or call fails.

### Agents

1. **Dataset Analyst** - Profiles dataset, detects types, missing values, outliers
2. **Workflow Planner** - Creates ML pipeline plan based on dataset characteristics
3. **Preprocessing Agent** - Builds sklearn pipelines (imputation, scaling, encoding)
4. **EDA Agent** - Generates meaningful visualizations and insights
5. **Feature Engineering Agent** - Conservative feature creation with experimental validation
6. **Model Selection Agent** - Chooses appropriate models based on data characteristics
7. **Hyperparameter Tuning Agent** - Optuna-based optimization with cross-validation
8. **Evaluation Agent** - Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC, etc.)
9. **Critic Agent** - Analyzes weaknesses, recommends actionable improvements
10. **Experiment Manager** - Tracks all experiments in MLflow + PostgreSQL
11. **Final Report Agent** - Generates comprehensive JSON/HTML reports

## Technology Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI, Python 3.12+ |
| Orchestration | LangGraph, LangChain |
| LLM | Groq (Llama 3.3 70B) |
| ML | scikit-learn, XGBoost, Optuna |
| Database | PostgreSQL (asyncpg + SQLAlchemy) |
| Experiment Tracking | MLflow |
| Visualization | matplotlib, seaborn |
| Frontend | React 18, Vite, Tailwind CSS |
| Deployment | Docker Compose |

## Project Structure

```
autonomous-ml-engineer-agent/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI routes
│   │   ├── agents/           # LangGraph agent nodes
│   │   ├── graph/            # State, workflow, edges
│   │   ├── tools/            # Deterministic ML tools
│   │   ├── ml/               # Core ML implementations
│   │   ├── models/           # Pydantic/SQLAlchemy models
│   │   ├── services/         # Business logic services
│   │   ├── config.py         # Settings management
│   │   ├── database.py       # Database connection
│   │   └── main.py           # FastAPI application
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── pages/            # Page components
│   │   ├── services/         # API services
│   │   └── hooks/            # Custom hooks
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── Dockerfile
├── datasets/                 # Sample datasets
├── mlruns/                   # MLflow artifacts
├── reports/                  # Generated reports
├── tests/                    # Unit/integration tests
├── docker-compose.yml
├── .gitignore
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- PostgreSQL 14+
- Docker & Docker Compose (optional)
- Groq API key

### Using Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd autonomous-ml-engineer-agent

# Set up environment variables
cp backend/.env.example backend/.env
# Edit backend/.env with your GROQ_API_KEY

# Start all services
docker-compose up -d

# Access the application
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
# MLflow: http://localhost:5000
```

### Local Development

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your settings
uvicorn app.main:app --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev

# PostgreSQL & MLflow (run separately or via docker-compose)
docker-compose up -d postgres mlflow
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key for LLM | Required |
| `LLM_MODEL` | LLM model name | `llama-3.3-70b-versatile` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://postgres:postgres@localhost:5432/autonomous_ml` |
| `MLFLOW_TRACKING_URI` | MLflow server URI | `http://localhost:5000` |
| `MAX_ITERATIONS` | Max autonomous iterations | `5` |
| `MIN_IMPROVEMENT_THRESHOLD` | Min improvement to continue | `0.01` |

## API Endpoints

### Datasets
- `POST /api/v1/datasets/upload` - Upload CSV dataset
- `GET /api/v1/datasets/{dataset_id}` - Get dataset info
- `GET /api/v1/datasets/` - List datasets

### Agent
- `POST /api/v1/agent/run` - Start autonomous ML workflow
- `POST /api/v1/agent/jobs` - Create new job
- `GET /api/v1/agent/jobs/{job_id}` - Get job status

### Experiments
- `GET /api/v1/experiments/` - List experiments
- `GET /api/v1/experiments/{experiment_id}` - Get experiment details
- `GET /api/v1/experiments/{experiment_id}/metrics` - Get experiment metrics

### Reports
- `GET /api/v1/reports/{job_id}` - Get final report (JSON)
- `GET /api/v1/reports/{job_id}/html` - Get final report (HTML)

## Usage Example

```bash
# 1. Upload dataset
curl -X POST "http://localhost:8000/api/v1/datasets/upload" \
  -F "file=@datasets/titanic.csv" \
  -F "target_column=survived" \
  -F "problem_type=auto"

# Response: {"job_id": "job_abc123", "dataset_id": "ds_xyz789"}

# 2. Start autonomous workflow
curl -X POST "http://localhost:8000/api/v1/agent/run" \
  -H "Content-Type: application/json" \
  -d '{"job_id": "job_abc123"}'

# 3. Monitor progress
curl "http://localhost:8000/api/v1/agent/jobs/job_abc123"

# 4. Get final report
curl "http://localhost:8000/api/v1/reports/job_abc123"
```

## Sample Datasets

The project includes sample datasets in `datasets/`:
- `sample_classification.csv` - Titanic-like dataset with missing values, categorical features, class imbalance
- `sample_regression.csv` - Housing prices dataset with numerical/categorical features

## Agentic Behavior Example

```
[12:31:01] Dataset Analyst: Dataset contains 7,043 rows, 12 columns
[12:31:02] Dataset Analyst: Missing values in Age (2.5%), Cabin (9.8%)
[12:31:03] Dataset Analyst: Problem type: Classification (binary)
[12:31:05] Preprocessing: Stratified split, imputation, scaling, encoding
[12:31:12] EDA: Strong Sex-Survival correlation, Fare-Pclass correlation
[12:31:15] Features: Created FamilySize, IsAlone, Title, Fare_log
[12:31:20] Models: XGBoost (0.87), RandomForest (0.85), LogisticRegression (0.82)
[12:31:25] Critic: Low recall (0.62) for positive class → recommends class_weight
[12:31:29] Improvement: Applied class_weight="balanced"
[12:31:35] Tuning: Optuna 50 trials → best params found
[12:31:45] Evaluation: New F1 = 0.91 (+0.04 improvement)
[12:31:50] Critic: Significant improvement achieved, stopping
[12:31:52] Report: Generated final report with all findings
```

## Testing

```bash
# Backend tests
cd backend
pytest tests/ -v --cov=app

# Frontend tests
cd frontend
npm test
```

## Development Phases

- ✅ Phase 1: Architecture + folder structure + requirements + configuration
- ✅ Phase 2: Dataset analysis tools
- ✅ Phase 3: ML preprocessing + EDA
- ✅ Phase 4: Model selection + training
- ✅ Phase 5: Optuna tuning
- ✅ Phase 6: MLflow experiment tracking
- ✅ Phase 7: LangGraph agents and state
- ✅ Phase 8: Critic + autonomous improvement loop
- ✅ Phase 9: FastAPI endpoints
- ✅ Phase 10: React dashboard
- ✅ Phase 11: PostgreSQL integration
- ✅ Phase 12: Docker
- ✅ Phase 13: Testing
- ✅ Phase 14: README + final cleanup

## Interview Talking Points

### 30-Second Pitch
> "I built an autonomous ML engineer agent that doesn't just run AutoML—it thinks like a data scientist. Using LangGraph, it observes results, critiques its own work, and iteratively improves models until convergence. It handles the full lifecycle: profiling, preprocessing, EDA, feature engineering, model selection, tuning, and reporting—all with MLflow tracking and a React dashboard."

### Key Differentiators from Traditional AutoML
1. **Critic Agent**: Identifies specific weaknesses (low recall, overfitting, imbalance) and maps to executable actions
2. **Autonomous Loop**: Continues until stopping criteria, not fixed pipeline
3. **LLM for Reasoning Only**: Python tools do computation; LLM plans and decides
4. **Full Experiment Lineage**: Every iteration tracked with parent-child relationships
5. **No Data Leakage**: Train/test split before any transformation fitting

## Future Improvements

- [ ] Multi-modal data support (images, text)
- [ ] Distributed training with Ray
- [ ] Advanced feature engineering (featuretools integration)
- [ ] Model explainability (SHAP integration in critic)
- [ ] A/B testing framework for deployment
- [ ] Kubernetes deployment manifests
- [ ] WebSocket/SSE for real-time agent logs

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

---

Built as a portfolio project demonstrating agentic AI, ML engineering, and full-stack development skills.