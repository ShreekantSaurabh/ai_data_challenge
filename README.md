# AI Marketing ROI Challenge

This repository contains an end-to-end workflow for identifying the most profitable customer groups for marketing campaigns. It covers exploratory data analysis (EDA), feature selection, model training with multiple tree-based learners, ROI simulation and a FastAPI prediction service that can be deployed locally or via Docker.

## Repository Structure
```text
ai_data_challenge/
├── customerGroups.csv
├── docker-compose.yml
├── Dockerfile
├── main.py
├── notebooks/
│   └── EDA_and_Model_Training.ipynb
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── data_loader.py
│   ├── model.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── logger.py
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   └── test_preprocessing.py
├── .gitignore
├── README.md
└── requirements.txt
```
- `customerGroups.csv` – raw labeled dataset used for training and evaluation.
- `main.py` – command-line entry point for running the full training pipeline.
- `src/` – project source code.
	- `data_loader.py` – handles CSV or GCS data loading.
	- `preprocessing.py` – imputes and scales features while preserving column names.
	- `model.py` – wraps model instantiation, class balancing, cross-validation and persistence.
	- `pipeline.py` – orchestrates end-to-end training and artifact saving.
	- `api.py` – FastAPI application exposing the `/features` and `/predict` endpoint.
	- `utils/` – logger and configuration helpers.
- `notebooks/EDA_and_Model_Training.ipynb` – interactive analysis, feature selection and model comparison.
- `models/` – default location for serialized model (`trained_model.pkl`) and preprocessor (`preprocessor.pkl`).
- `tests/` – automated tests for the API, preprocessing and model modules.
- `Dockerfile` / `docker-compose.yml` – containerized deployment assets.
- `requirements.txt` – Python dependencies.

## Prerequisites
- Python 3.10
- Git (optional).
- For Docker usage: Docker Engine 24+ and Docker Compose.
- Recommended: a virtual environment tool (e.g., `python -m venv venv`).

## Local Environment Setup (Windows Command Prompt)
```cmd
cd C:\Users\<your-user>\OneDrive\Desktop\ai_data_challenge
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If you prefer to override default paths or model selection, set the environment variables documented in `src/utils/config.py` before running training or serving (e.g., `set MODEL_TYPE=ada`).

## Running the Training Pipeline
The pipeline loads `customerGroups.csv`, performs preprocessing, cross-validates the desired model, fits on all data and saves artifacts under `models/`.

```cmd
# Activate your virtualenv first
python main.py --model-type ada
```

CLI options:
- `--model-type` selects the estimator (`dummy`, `rf`, `gbm`, `xgb`, `cat`, `ada`, `tree_ensemble`; default `ada`).
- `--data-path` overrides the training dataset (default `customerGroups.csv`).
- `--model-path` / `--prep-path` control where the trained model and preprocessor are saved; parent directories are created automatically.

Artifacts are written to:
- `models/trained_model.pkl` – the fitted estimator.
- `models/preprocessor.pkl` – the fitted `Preprocessor` object that retains feature names.

Training logs are emitted to stdout via the custom logger.

## Exploring the Notebook
Open `notebooks/EDA_and_Model_Training.ipynb` in VS Code or JupyterLab. The notebook documents:
- Detailed EDA (missingness, mutual information).
- Feature Selection.
- Model benchmarking (Dummy, RandomForest, AdaBoost, GradientBoosting, XGBoost, CatBoost, soft voting ensemble tree).
- ROI uplift computation.

Run cells sequentially to reproduce charts and tables. The notebook relies on the same Python environment; ensure dependencies are installed before executing.

## Serving Predictions Locally (FastAPI)
After training using main.py (or running EDA_and_Model_Training.ipynb), the API can load the artifacts and serve predictions from `models/`.

```cmd
venv\Scripts\activate
uvicorn src.api:app --host 0.0.0.0 --port 8080
```

Test the endpoint with a sample payload in browser:
Verify the server is up: Visit http://127.0.0.1:8080/docs in browser. 

Click on GET /features --> “Try it out" --> "Execute". It gives the response as the exact column or feature order the trained model expects.

Click on POST /predict --> “Try it out” --> paste a JSON object with expected feature and execute; the response pane shows the prediction. If a feature is not provided then it will considered as 0. If any unrequired feature is provided then it will be ignored.
JSON example:
{
  "g1_1": 0.27, "g1_5": 0.43, "g2_1": 0.35, "g2_11": 0.18, "g2_19": 0.22, "c_2": 0.41, "c_3": 0.33, "c_6": 0.57, "c_10": 0.29, "c_11": 0.21, "c_25": 0.14
}

Test the endpoint with a sample payload in other cmd window:
```cmd
curl -X POST http://127.0.0.1:8080/predict -H "Content-Type: application/json" -d "{\"g1_1\":0.27,\"g1_5\":0.43,\"g2_1\":0.35,\"g2_11\":0.18,\"g2_19\":0.22,\"c_2\":0.41,\"c_3\":0.33,\"c_6\":0.57,\"c_10\":0.29,\"c_11\":0.21,\"c_25\":0.14}"
```
The response includes the predicted profitability label (`none profitable`, `group 1`, or `group 2`). 

Example Response: {"predicted_target":"group 1"}

## Running Tests
Use pytest for the automated checks. Ensure the virtual environment is activated so dependencies (including FastAPI and httpx) are available.

```cmd
python -m pytest -v
```

Individual files can be targeted, for example:
```cmd
python -m pytest tests/test_model.py
python -m pytest tests/test_preprocessing.py
python -m pytest tests/test_api.py
```

## Container Usage (Docker or Podman)
The project ships with standard OCI container assets. Replace `docker` with `podman` if you prefer Podman; the commands are drop-in compatible.

### Build and run directly
```cmd
docker build -t ai-data-challenge .
docker run --rm -p 8080:8080 ai-data-challenge
```

Podman users can execute the same workflow:
```cmd
podman build -t ai-data-challenge .
podman run --rm -p 8080:8080 ai-data-challenge
```

### Using Compose
```cmd
docker compose up --build
```

For Podman users:
```cmd
podman compose up --build
```

Test the endpoint as we have done earlier above.

Both approaches expose the FastAPI service on port `8080`. The container executes `uvicorn src.api:app` and automatically installs dependencies from `requirements.txt`. Copy or mount `customerGroups.csv` if you want to retrain inside the container; otherwise, package pre-trained artifacts.

## Checking Outputs & Logs
- **Model metrics:** Printed during notebook execution and CLI training (cross-validation F1, accuracy, confusion matrix).
- **Artifacts:** Stored under `models/`. Inspect with `joblib.load("models/trained_model.pkl")` in an interactive shell if needed.
- **Logs:** Console output from training includes data loading, preprocessing status, and final save confirmation. The API logs requests through Uvicorn.
- **ROI uplift & insights:** Generated in the notebook (Section 8) and printed in the training summary cells.

## Configuration & Extensions
- Override defaults by editing `src/utils/config.py` or setting environment variables (`DATA_PATH`, `MODEL_PATH`, `PREP_PATH`, `MODEL_TYPE`, `GCS_URI`).
- Extend `CampaignModel` in `src/model.py` to register additional algorithms or tuning strategies.
- Add new engineered features in the notebook or inside `pipeline.py` before preprocessing.
- For GCS-sourced data, ensure `gcsfs` credentials are configured and set `DATA_PATH` to the GCS URI.

## Troubleshooting
- **`ModuleNotFoundError: src`:** Ensure the project root is on `PYTHONPATH`. Running via `python -m pytest` or `python main.py` from the repo root avoids this issue.
- **Gradient boosting considerations:** Ensure the `Preprocessor` preserves column order so feature alignment remains consistent for scikit-learn's gradient boosting.
- **FastAPI test failures:** Install `httpx` (included in `requirements.txt`) and rerun tests.
- **Docker build failures:** Verify internet access and that the dataset is present within the build context.

## Suggested Workflow for New Contributors
1. Clone repository and set up the virtual environment.
2. Review the notebook to understand data characteristics and current metrics.
3. Run `python main.py` to reproduce baseline artifacts.
4. Launch the FastAPI service (`uvicorn` locally or Docker) and test predictions with sample payloads.
5. Enhance features or models in the notebook/pipeline, re-run training, and document findings.
6. Update tests or add new ones when extending functionality.


## Technical Architecture Diagram

---
config:
  layout: elk
---
flowchart TD
    DATA["CustomerGroups.csv<br>or GCS Data Source"] --> PIPELINE["Training Pipeline (main.py)<br>Preprocessing + Model Training"]
    NOTEBOOK["EDA &amp; Model Training Notebook<br>(Exploration, Benchmarking, ROI Uplift)"] --> PIPELINE
    PIPELINE --> MODEL["ML Model Artifacts<br>• preprocessor.pkl<br>• trained_model.pkl"]
    MODEL --> STORAGE["Google Cloud Storage<br>(Artifact Repository)"]
    STORAGE --> API["FastAPI Prediction API<br>(Uvicorn, Dockerized)"]
    API --> CLOUDRUN["GCP Cloud Run<br>(Container Hosting)"] & BIGQUERY["BigQuery<br>(Logs / Campaign History)"]
    UI["Marketing Dashboard / Streamlit UI"] -- POST /predict --> CLOUDRUN
    CLOUDRUN --> API & MONITOR["Cloud Logging / Monitoring"]
    API -- Load model + preprocessor --> MODEL
    MONITOR --> PIPELINE
     UI:::ui
     API:::api
     PIPELINE:::ml
     MODEL:::data
     DATA:::data
     NOTEBOOK:::ml
     CLOUDRUN:::infra
     STORAGE:::data
     BIGQUERY:::data
     MONITOR:::infra
    classDef ui fill:#E3F2FD,stroke:#1E88E5,color:#0D47A1,fontWeight:bold
    classDef api fill:#E8F5E9,stroke:#43A047,color:#1B5E20,fontWeight:bold
    classDef ml fill:#FFF3E0,stroke:#FB8C00,color:#E65100,fontWeight:bold
    classDef data fill:#F3E5F5,stroke:#8E24AA,color:#4A148C,fontWeight:bold
    classDef infra fill:#ECEFF1,stroke:#546E7A,color:#263238,fontWeight:bold


## Sequence Diagram

sequenceDiagram
    autonumber
    participant U as User (Marketing Dashboard)
    participant F as FastAPI Service (Cloud Run)
    participant M as ML Engine (Preprocessor + Model)
    participant S as GCS Storage (Artifacts)
    participant B as BigQuery (Logs / History)

    U->>F: 1. POST /predict (JSON: features of group 1 & group 2)
    F->>S: 2. Retrieve latest model & preprocessor (pkl files)
    S-->>F: 3. Return serialized artifacts
    F->>M: 4. Send cleaned features to ML Engine
    M->>M: 5. Apply preprocessing (imputation, scaling)
    M->>M: 6. Run model.predict() to infer best group
    M-->>F: 7. Return predicted target (0, 1, or 2)
    F->>B: 8. Log request, prediction, timestamp
    F-->>U: 9. Respond { "predicted_target": "group 1" }
    Note right of F: Logs and predictions stored<br/>for monitoring & retraining

