# Workflow-CI

```
Workflow-CI/
├── .github/workflows/ci.yml
├── MLProject/
│   ├── MLproject
│   ├── conda.yaml
│   ├── modelling.py
│   ├── namadataset_preprocessing/
│   │   ├── train.csv
│   │   └── test.csv
│   └── DockerHub.txt
└── README.md
```

## Run locally

```bash
cd MLProject
mlflow run . --env-manager=local --experiment-name telco_churn_ci
```
