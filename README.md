# AI in Physiotherapy: Automated Pain Assessment Using Physiological Signals

## 1. Project Overview

### 1.1 Project Vision
The Automated Pain Assessment (APA) System is a proof-of-concept designed to predict a standardized, objective, continuous pain rating on a 0–10 scale. It utilizes multimodal physiological signals—Blood Volume Pulse (BVP), Electromyography (EMG), Electrodermal Activity (EDA), and Respiration (RESP)—from the **PainMonit Dataset**.

The system leverages a hybrid data processing strategy, synchronizing pre-processed physiological data (`x.npy`) with continuous pain labels from raw CSV files (NRS for PMED, CoVAS for PMCD). This approach allows for a regression-based analysis to predict pain on a continuous 0-10 scale, which aligns with the standard Numerical Rating Scale (NRS) used in physiotherapy.

The ultimate goal is to provide physiotherapists with a tool for remote, data-driven pain assessment, improving diagnostic accuracy and enabling personalized treatment planning.

### 1.2 Key Objectives
- **Framework Design**: Develop a modular framework to parse and synchronize BVP, EMG, EDA, and RESP signals with continuous pain labels.
- **Feature Engineering**: Implement a comprehensive feature engineering pipeline to extract statistical, temporal, and frequency-domain features.
- **Machine Learning Model**: Train and evaluate a Gradient Boosting Trees (GBT) model for continuous pain prediction, using Leave-One-Subject-Out (LOSO) cross-validation.
- **Performance**: Achieve a Mean Absolute Error (MAE) of less than 1.5 on the 0-10 pain scale.
- **Explainability**: Use Explainable AI (XAI) techniques, such as feature importance analysis from tree-based models, to provide insights into the key physiological drivers of pain.(if time permits)

## 2. System Architecture & Methodology

The system is designed as a modular pipeline with distinct layers for each processing stage:

1.  **Data Alignment (`data_alignment`)**: Loads physiological data (`x.npy`), subject data (`subjects.npy`), and ground truth pain labels (raw `.csv` files) and implements a critical timestamp-based algorithm to align the physiological data windows with the continuous pain ratings.
3.  **Feature Engineering (`feature_engineering`)**: Extracts a rich set of features from the synchronized signals and selects the most predictive ones using methods like Pearson correlation, PCA, and model-based feature importance.
4.  **Modeling (`model_development`)**: Trains a Gradient Boosting Regressor on the engineered features to predict pain scores.
5.  **Evaluation**: Assesses model performance using LOSO cross-validation to ensure subject-independent generalization.

![Flowchart](docs/flowchart/graph (13).png)

## 3. Dataset

This project uses the [PainMonit Dataset](https://www.nature.com/articles/s41597-024-03862-7), which includes two key subsets:
- **PMED**: An experimental dataset collected in a controlled lab setting.
- **PMCD**: A clinical dataset collected from patients in a real-world clinical environment.

The use of both datasets allows for the analysis of the "lab-to-clinic" gap and evaluates the model's robustness in different contexts.

## 4. Project Structure

```
AI_Physiotherapy_Pain_Assessment/
├── data/
│   ├── raw/
│   │   └── PMD/         # PainMonit Dataset
│   └── processed/       # Processed data after pipeline execution
├── docs/
│   ├── dissertation/
│   └── flowchart/
├── src/
│   ├── data_alignment/
│   ├── feature_engineering/
│   ├── model_development/
│   └── pain_prediction/
├── models/              # Saved model artifacts
├── results/             # Evaluation results and plots
└── README.md
```

## 5. Getting Started

### Prerequisites
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Usage
1.  **Data Setup**: Place the PainMonit dataset into the `data/raw/PMD/` directory.
2.  **Run Pipeline**: Execute the main preprocessing and training script (to be created).
    ```bash
    python run_pipeline.py
    ```

## 6. Future Enhancements
The modular design of this project allows for several future extensions:
- **Advanced XAI**: Integrate methods like SHAP or LIME for deeper model interpretability.
- **NLP Integration**: Incorporate patient narratives to add another modality to the pain assessment.
- **EEG Analysis**: Add EEG signal processing for emotion classification to distinguish between physical and emotional pain.
- **Patient Profiling**: Develop longitudinal patient profiles to track pain over time.

---
*This project is for academic purposes as part of an MSc in Artificial Intelligence.*
*Author: Wing Kiu Lo*
*Date: July 2025*
