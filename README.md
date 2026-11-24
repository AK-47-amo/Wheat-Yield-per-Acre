# Group G214 --- MP130 Wheat Yield per Acre (Patna)

## Problem Overview
- **Group ID**: G214
- **Problem ID**: MP130
- **Client**: Patna Kisan Producer Company
- **Target**: `yield_qtl_per_acre` (Quintals per acre)
- **Metric**: MAE (Mean Absolute Error)

## Group Members
- S24CSEU0978 - Aditya Kumar
- S24CSEU0993 - Shradhey Mishra
- S24CSEU0976 - Kanak Kapil
- S24CSEU0313 - Ramita Gaur
- S24CSEU0977 - Arishita

## Architecture & Repository Structure
- `data/`: Raw data tracked via DVC (`train.csv.dvc`, `later.csv.dvc`) and holdout features
- `src/wheat_yield/`: Source code package split into data, train, drift, and predict modules
- `params.yaml`: Centralized configuration loaded via `yaml.safe_load`
- `mlflow.db`: SQLite database tracking experiments and model registry (`@champion` alias)
- `models/`: Exported champion model (`model.joblib`)
- `app/`: FastAPI prediction service with Pydantic validation rules and structured JSON logging
- `tests/`: Pytest automated test suite including TestClient tests
- `Dockerfile` & `k8s/`: Containerization and Kubernetes Deployment / Service manifests
- `.github/workflows/ci.yml`: Continuous integration workflow running tests on push
- `reports/findings.yaml`: Audit report covering data issues, MLflow tracking, and monitoring
- `predictions/holdout_predictions.csv`: Model predictions on `holdout_features.csv`
