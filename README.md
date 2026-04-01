# 🎓 Teacher Readiness Index (TRI)
### Modelling Teacher Readiness for Competency-Based Education: A Machine Learning Approach in Nakuru County, Kenya

> **MSc Thesis** | Peter Kanyuira Wachugu | Strathmore University, Institute of Mathematical Sciences | March 2026  
> Supervisor: Dr. John Olukuru

---

## 📌 Overview

Over 5,000 Kenyan secondary schools received **no Grade 10 applications in 2024** — a visible consequence of widespread doubt about teacher preparedness for the Competency-Based Curriculum (CBC). This project addresses that crisis directly.

The **Teacher Readiness Index (TRI)** is a psychometrically validated, machine-learning-powered instrument that measures and predicts CBC implementation readiness among secondary school teachers. Built on the **TPACK framework** and the **CRISP-DM process model**, it produces actionable diagnostic profiles at both the individual teacher and school level — delivered through a Streamlit dashboard.

---

## 🏗️ Project Architecture

```
TRI-Project/
│
├── 📊 Data (Raw & Processed)
│   ├── google_forms_teacher_survey_raw.csv        # 1,200 teachers × 46 items
│   ├── google_forms_principal_survey_raw.csv      # 100 principals × 15 items
│   ├── teacher_survey_cleaned_keyvars.csv         # Output of Notebook 01
│   ├── teacher_construct_scores.csv               # Output of Notebook 02
│   ├── teacher_readiness_train.csv                # Output of Notebook 03 (80%)
│   └── teacher_readiness_test.csv                 # Output of Notebook 03 (20%)
│
├── 📓 Notebooks (8 sequential analysis stages)
│   ├── 01_data_loading_eda.ipynb
│   ├── 02_psychometric_validation.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_modelling.ipynb
│   ├── 05_advanced_modelling.ipynb
│   ├── 06_stacking_ensemble.ipynb
│   ├── 07_final_evaluation.ipynb
│   └── 08_fairness_robustness.ipynb
│
├── 🤖 Model Outputs
│   └── outputs/
│       ├── notebook_04_BASELINE_MODELS/
│       ├── notebook_05_ADVANCED_MODELS/
│       ├── notebook_06_STACKING_ENSEMBLE/
│       └── notebook_07_MODEL_INTERPRETATION/
│           └── models/07_final_stacking_model.pkl  ← Champion model
│
├── 🖥️ Streamlit Application
│   └── app.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 The TRI Framework

The index is built from **28 validated survey items** across four constructs grounded in the TPACK framework:

| Construct | Items | Cronbach's α | Mean Score |
|---|---|---|---|
| Pedagogical Confidence | 7 | 0.962 | 3.19 / 5 |
| Digital Literacy | 9 | 0.952 | 2.66 / 5 |
| Resource Availability | 6 | 0.947 | 2.85 / 5 |
| Training Quality | 6 | 0.976 | 2.76 / 5 |

**TRI = Mean(Pedagogical + Digital + Resource + Training) × 4**  
Scale: **4 – 20** | Mean: **11.46** | Std: **1.83**

---

## 📓 Notebook Pipeline

| # | Notebook | What It Does | Key Output |
|---|---|---|---|
| 01 | Data Loading & EDA | Loads both survey CSVs from GitHub, cleans missing data (logical imputation for non-attendees), explores demographics and challenges | `teacher_survey_cleaned_keyvars.csv` |
| 02 | Psychometric Validation | EFA (parallel analysis → 4 factors), CFA on holdout (CFI=0.996, RMSEA=0.017), reliability analysis, scalar invariance across gender | `teacher_construct_scores.csv` |
| 03 | Feature Engineering | Rejects `overall_readiness` (variance = 0.25), constructs TRI, one-hot encoding, RobustScaler, challenge clusters, GroupShuffleSplit | `teacher_readiness_train/test.csv` |
| 04 | Baseline Modelling | Dummy → MLR → Lasso → Ridge → Decision Tree; GroupKFold CV; MLR achieves R²=0.9699 | `.pkl` models + tables |
| 05 | Advanced Modelling | Random Forest, Gradient Boosting, XGBoost, SVR; none beat MLR | `.pkl` models |
| 06 | Stacking Ensemble | Stacks MLR + Ridge + XGBoost + SVR under Ridge meta-learner; R²=0.9706 CV | `06_stacking_ensemble.pkl` |
| 07 | Final Evaluation | Single test-set reveal on 97 unseen teachers from 3 unseen schools; SHAP analysis; residual diagnostics | R²=0.9769, MAE=0.245 |
| 08 | Fairness & Robustness | Subgroup audits by school type and location; R²>0.95 across all groups | Fairness verdict: ✅ |

---

## 🤖 Model Results

### Performance Hierarchy

| Model | CV R² | Test R² | RMSE |
|---|---|---|---|
| Dummy (Mean) | -0.086 | — | 1.858 |
| Decision Tree | 0.730 | — | 0.925 |
| Random Forest | 0.907 | — | — |
| XGBoost | 0.958 | — | — |
| SVR | 0.970 | — | — |
| **MLR (Baseline)** | **0.9699** | **0.9744** | **0.308** |
| **Stacking Ensemble ✅** | **0.9706** | **0.9769** | **0.307** |

### Champion Model: Stacking Ensemble

```
Level 1 Base Learners:
  ├── MLR       (weight: 16.5%) — linear foundation
  ├── Ridge     (weight: 18.3%) — stability & regularisation  
  ├── XGBoost   (weight: 27.7%) — non-linear residuals
  └── SVR       (weight: 38.5%) — boundary case detection

Level 2 Meta-Learner: Ridge Regression (α = 1.0)
Validation: 5-fold GroupKFold (school-clustered, zero leakage)
```

### Test Set Performance (97 unseen teachers, 3 unseen schools)

- **R² = 0.9769**
- **MAE = 0.245 points**
- **MAPE = 2.17%**
- **87.6%** of predictions within ±0.5 points
- **100%** of predictions within ±1.0 points

---

## 🔍 Key Findings

### 1. Pedagogical Confidence Dominates
Self-efficacy (β = 0.461 for classroom management, β = 0.452 for learner engagement) accounts for ~50% of predictive power — consistent with Bandura's Self-Efficacy Theory.

### 2. Training Quality > Attendance
Training attendance was **not statistically significant** (p = 0.364). Individual training quality items (β ≈ 0.14–0.21) are the real drivers. Implication: invest in training content, not just headcounts.

### 3. The Digital-Training Threshold Interaction
Digital Literacy only predicts readiness when Training Quality **exceeds 3.0** on the 5-point scale. Below this threshold, digital upskilling has limited impact — a non-linear pattern that standard regression cannot detect.

### 4. Urban-Rural Digital Gap
Urban teachers score ~0.82 points higher on Digital Literacy than rural counterparts (digi_C1a–C2e differences = 0.66–1.01). This is a **substantive population difference**, not a measurement artifact — confirmed by invariance analysis.

### 5. The Model is Fair
| Subgroup | R² | MAE |
|---|---|---|
| Public schools (n=72) | 0.9778 | 0.234 |
| Private schools (n=25) | 0.9599 | 0.276 |
| Overall | 0.9769 | 0.245 |

No group suffers from disadvantageous underprediction. The Private school gap is attributable to small sample size (n=25).

---

## 🖥️ Streamlit Diagnostic Dashboard

The app provides five integrated tools:

| Tab | Function |
|---|---|
| 📝 Individual Assessment | 19-item diagnostic → TRI score + readiness band + radar chart + personalised recommendations |
| 🎯 Intervention Simulator | Adjust construct targets → predict TRI improvement before committing resources |
| 📊 Cohort Analysis | Upload CSV of multiple teachers → school-level TRI distribution + readiness band breakdown |
| 📈 Progress Tracking | Track score changes across multiple assessments within a session |
| ℹ️ Technical Documentation | Model performance, feature importance, key findings, references |

### Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Nyuira/TRI-Project.git
cd TRI-Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ensure the model file is present
# Place 07_final_stacking_model.pkl in the root directory

# 4. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Deploy on Streamlit Community Cloud (Free)

1. Push your repo to GitHub (including `app.py`, `requirements.txt`, and `07_final_stacking_model.pkl`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select `Nyuira/TRI-Project` → set main file to `app.py`
4. Click **Deploy** — your app will be live at `https://nyuira-tri-project.streamlit.app`

---

## ⚙️ Setup & Requirements

```bash
pip install -r requirements.txt
```

**Core dependencies:**

```
pandas>=1.5
numpy>=1.23
matplotlib
seaborn
scikit-learn>=1.2
factor-analyzer
semopy
pingouin
scipy
statsmodels
xgboost
joblib
requests
streamlit>=1.28
plotly
```

**Environment:** Python 3.10+ | Google Colab (notebooks) | VSCode (Streamlit app)

---

## 📐 Methodology Summary

| Phase | Approach |
|---|---|
| **Framework** | TPACK + Bandura Self-Efficacy + Fullan Change Theory |
| **Process Model** | CRISP-DM (8 notebooks map to CRISP-DM phases) |
| **Sample** | 1,200 teachers × 240 schools, Nakuru County |
| **Survey Design** | 28-item Likert scale (1–5) across 4 constructs |
| **Validation** | EFA (calibration 70%) → CFA (holdout 30%) |
| **Split Strategy** | GroupShuffleSplit by school cluster (zero leakage) |
| **CV Strategy** | 5-fold GroupKFold (school-level independence) |
| **Interpretability** | MLR coefficients + SHAP analysis |
| **Fairness** | Subgroup audit by school type and location |

---

## 📁 Data Notes

- **Raw data** is simulated survey data representative of Nakuru County secondary schools
- Both CSVs are loaded directly from this GitHub repo via raw URLs in each notebook
- Processed outputs (`teacher_survey_cleaned_keyvars.csv`, `teacher_construct_scores.csv`, `teacher_readiness_train.csv`, `teacher_readiness_test.csv`) must also be present in the repo root for notebooks to run sequentially
- The `.pkl` model file (`07_final_stacking_model.pkl`) must be in the **root directory** for the Streamlit app to load

---

## 🔒 Ethics & Privacy

- All survey responses are anonymised; respondent IDs are synthetic
- No personally identifiable information is stored or transmitted
- The diagnostic dashboard does not log or retain individual assessment data beyond the active session
- Research approved under Strathmore University ethical guidelines

---

## 📖 Citation

```bibtex
@mastersthesis{wachugu2026tri,
  author    = {Peter Kanyuira Wachugu},
  title     = {Modelling Teacher Readiness for Competency-Based Education: 
               A Machine Learning Approach in Nakuru County, Kenya},
  school    = {Strathmore University},
  year      = {2026},
  month     = {March},
  address   = {Nairobi, Kenya},
  note      = {MSc in Statistical Science}
}
```

---

## 👤 Author

**Peter Kanyuira Wachugu**  
MSc Statistical Science, Strathmore University  
📧 research@strathmore.edu  
🔗 [github.com/Nyuira](https://github.com/Nyuira)

Supervisor: **Dr. John Olukuru**, Institute of Mathematical Sciences, Strathmore University

---

*© 2026 Strathmore University. This repository is available for academic use with proper acknowledgement.*