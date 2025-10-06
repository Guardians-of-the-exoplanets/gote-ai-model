# Guardians of The Exoplanets - AI Model - Single Prediction, Batch Prediction, Automated Retrain and Cloud Web Server

[Our Team](https://www.spaceappschallenge.org/2025/find-a-team/guardians-of-the-exoplanets/?tab=project)

# File Structure
```
.
├── api_server – Core API backend for handling requests and serving models. This server is running on Cloud Run (Google Cloud Platform) and is used by our backend project/
│   ├── data – Stores datasets, organized by mission/
│   │   └── k2 / kepler / tess – Separate folders for each mission, each containing:/
│   │       ├── X_blind.csv – Features for blind evaluation (unlabeled)
│   │       ├── X_test.csv – Features for the test set
│   │       ├── X_train_full.csv – Features for the full training set
│   │       ├── y_blind.csv – Labels for blind evaluation
│   │       ├── y_test.csv – Labels for the test set
│   │       └── y_train_full.csv – Labels for the full training set
│   ├── functions.py – Utility functions for data processing or model operations
│   ├── main.py – API server entry point
│   └── requirements.txt – Python dependencies for the API server
├── models – Pretrained models for each mission/
│   ├── *_complete.model – Full-featured model
│   └── *_lite.model – Lightweight / simplified model for faster inference
├── notebooks – Jupyter notebooks for training, evaluation, and experimentation/
│   ├── *_complete.ipynb – Experiments with full-featured models
│   └── *_light.ipynb – Experiments with lightweight models
├── schemas – Feature definitions or column schemas/
│   ├── *_filtered.txt – Subset of relevant features
│   └── *_full.txt – All available features
└── summary – CSV reports summarizing model feature importance/
    ├── *_all_feature_importance.csv – Full feature importance scores
    └── *_feature_importance_summary.csv – Aggregated / summarized importance per feature set
```


