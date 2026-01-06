# 🛰️ Satellite Imagery–Enhanced Property Valuation

**CDC × Yhills Open Projects (2025–26)**  
**Project type:** Multimodal regression for residential property price prediction  
**Author:** Sachin Meena  
**Tech:** Python · PyTorch · scikit-learn


---

## 📌 Table of contents

- Overview  
- Problem statement  
- Why satellite imagery?  
- Pipeline (diagram + code block)  
- Pipeline steps (detailed)  
- Data & preprocessing (including image notes)  
- Models implemented  
- Results (metrics table)  
- Explainability (Grad-CAM observations)  
- Project layout (files & notebooks)  
- Setup & installation  
- How to run / execution order  
- Producing predictions & outputs  
- Limitations & future work  
- Contributing & contact  
- License

---

## 🔍 Overview

This project explores whether satellite imagery — providing neighborhood-level visual context — can improve residential property price prediction when combined with traditional tabular housing features (area, rooms, condition, etc.). The work implements three modelling strategies (tabular-only, image-only, multimodal fusion), evaluates them on standard metrics, and uses Grad-CAM to interpret image influence.

---

## ❓ Problem statement

Standard property valuation models (AVMs) often rely on structured features but miss environmental neighborhood factors (green space, water access, road layout, density). The aim is to build a robust multimodal pipeline that:

- gathers satellite tiles for each property coordinate,
- extracts visual embeddings with a pretrained CNN,
- integrates image embeddings with structured data,
- trains/evaluates models and interprets image influence.

**Research question:** Can satellite imagery improve price prediction accuracy when fused with tabular features?


## 🛰️ Why satellite imagery?

Satellite tiles can reveal:
- proximity to water and coastlines  
- green cover vs. built density  
- road networks and accessibility  
- urban layout and lot organization

These cues are complementary to internal property attributes and can be especially useful where local amenities or environmental factors drive price differences.

---

## 🔁 Pipeline (visual + code-style)

### Monospace pipeline (suitable for README)
```text
Property Location (Lat/Lng)
            |
     ┌──────┴──────┐
     |             |
     v             v
Satellite API   Tabular Data
     |        (sqft, beds, year, etc.)
     v             |
 CNN Feature Extractor |
    (ResNet-18, frozen)|
     |                 |
     └──────┬──────────┘
            |
            v
      Fusion Models
            |
            v
     Price Prediction

flowchart TB
  A[Property Location (Lat/Lng)]
  A --> B[Satellite API]
  A --> C[Tabular Data<br/>(sqft, beds, year, ...)]
  B --> D[ResNet-18 → CNN embeddings]
  C --> E[Tabular features → encoder]
  D --> F[Fusion / MLP]
  E --> F
  F --> G[Price prediction]
```

## ⚙️ Pipeline Steps (Detailed)

### 1. Data Ingestion
- Load the tabular housing dataset containing:
  - Sale price (target)
  - Property attributes such as `sqft_living`, `bedrooms`, `bathrooms`,
    `year_built`, `latitude`, `longitude`, `waterfront`, `view`, etc.

---

### 2. Sampling & Stratification
- Properties may be **stratified based on log(price)** to ensure satellite images
  are collected across the full price distribution (low, medium, and high-value homes).

---

### 3. Satellite Imagery Acquisition
- For each property, satellite tiles are fetched using `(latitude, longitude)`
  from a map provider (e.g., Mapbox, Google Static Maps).
- Images are collected at a **fixed zoom level and resolution**.
- Downloaded images are cached locally in:
- Metadata such as zoom level and timestamp is stored alongside image paths.

> **Note:** Satellite images are excluded from version control due to
> storage constraints and API usage limits.

---

### 4. 📂 Data & Preprocessing Details

#### Tabular Data
- Handle missing values using **median imputation** for numerical features.
- Apply **log transformation** to the target variable (price).
- Scale numerical features using standardization.
- Encode categorical variables using **one-hot** or **target encoding**.

#### Image Data
- Resize images to **224 × 224** pixels.
- Normalize pixel values according to the pretrained CNN requirements.

### Tabular Features (Examples)
- `sqft_living`, `sqft_lot`
- `bedrooms`, `bathrooms`
- `condition`, `grade`
- `year_built`
- `lat`, `long`
- `sqft_living15`, `sqft_lot15`
- `waterfront`, `view`

### Target Variable
- Sale price (log-transformed during training)

### Image Acquisition Notes
- Satellite tiles collected at a consistent zoom level.
- Mapping from `property_id → image_path` stored in CSV format.
- Example images for **low-priced** and **high-priced** areas are included
in the project report (`23116085_report.pdf`).

---

### 5. Feature Extraction
- Use a pretrained **ResNet-18** model with the classification head removed.
- Extract **512-dimensional image embeddings**.
- CNN weights are initially **frozen** to reduce overfitting and computation cost.

---

### 6. Modeling

- **Tabular-only model**
- Gradient boosting (LightGBM / XGBoost)
- Uses only structured housing features

- **Image-only model**
- MLP trained on CNN image embeddings
- Evaluates predictive power of satellite imagery alone

- **Multimodal (Early Fusion)**
- Concatenate standardized tabular features with image embeddings
- Pass combined features through an MLP regressor

## 🧪 Models Implemented

### 1️⃣ Tabular Baseline
- **Model:** Gradient Boosting Regressor
- **Purpose:** Performance benchmark
- **Observation:** Best performing approach


### 2️⃣ Image-Only Model
- **Backbone:** Pretrained ResNet-18
- **Architecture:** CNN embeddings → MLP
- **Purpose:** Measure standalone predictive signal from imagery


### 3️⃣ Multimodal Fusion Model
- **Strategy:** Early fusion (feature concatenation)
- **Architecture:** Tabular features + image embeddings → MLP
- **Note:** Simple concatenation was used; more advanced fusion techniques
(attention-based or late-fusion) are recommended for future work.
## 📈 Results

**Evaluation Metrics**
- **RMSE** computed on the original price scale  
- **R²** score to measure explained variance  
---

### 7. Training & Evaluation
- **Loss Function:** Mean Squared Error (MSE) on log(price)
- **Evaluation Metrics:**
- RMSE (converted back to original price scale)
- R² score
- **Validation Strategy:**
- Holdout split or k-fold cross-validation
- Spatial grouping recommended to reduce geographic leakage

---

### 8. Explainability
- Apply **Grad-CAM** to CNN feature maps.
- Overlay heatmaps on satellite images to visualize regions influencing predictions.
- Helps verify whether the model focuses on meaningful neighborhood features.

---

### Model Performance

| Model | RMSE | R² |
|------|------:|----:|
| Tabular-Only | 88,521.20 | 0.9376 |
| Image-Only | 539,792.00 | -1.2756 |
| Multimodal (Early Fusion) | 489,662.99 | -0.6775 |

### Key Takeaways
- **Tabular-only model** achieves the best performance, indicating that structured housing attributes provide the strongest predictive signal.
- **Image-only model** captures some visual information but performs poorly in terms of raw price prediction accuracy.
- **Early multimodal fusion** does not improve performance; simple feature concatenation likely introduces noise rather than complementary information.

  ## 🛠️ Setup

### Prerequisites
- Python **3.8** or higher  
- Git  

---

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/satellite-property-valuation.git
cd satellite-property-valuation





