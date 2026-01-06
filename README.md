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

---

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

⚙ Pipeline steps (detailed)

Data ingestion

Load tabular dataset containing sale price and property features (sqft, bedrooms, bathrooms, year built, lat/long, waterfront, view, etc.).

Sampling & stratification

Optionally stratify properties by log(price) to collect image tiles across full price distribution.

Satellite imagery acquisition

Given (lat, lon) fetch tiles at a fixed zoom/size from a provider (Mapbox, Google Static Maps, or other).

Cache images locally (data/images/) and store metadata (zoom, timestamp).

Note: images are excluded from repo due to size & API limits.

Preprocessing

Tabular: impute, log-transform price target, scale numeric features, encode categoricals.

Images: resize (224×224), normalize per pretrained network spec.

Feature extraction

Use pretrained ResNet-18 (remove classification head) to get 512-d embeddings. Initially freeze weights.

Modeling

Tabular-only: gradient boosting (LightGBM / XGBoost) on structured features.

Image-only: MLP on CNN embeddings.

Multimodal (early-fusion): concatenate standardized tabular features + image embeddings → MLP regressor.

Training & evaluation

Loss: MSE on log(price).

Metrics: RMSE (original price scale), R².

Validation: k-fold or holdout; consider spatial grouping to avoid leakage.

Explainability

Apply Grad-CAM to CNN activations and overlay heatmaps on tiles to inspect model attention.

📂 Data & preprocessing

Tabular features (examples): sqft_living, sqft_lot, bedrooms, bathrooms, condition, grade, year_built, lat, long, sqft_living15, sqft_lot15, waterfront, view.

Target: sale price (log-transformed for modeling).

Image acquisition notes: collect tiles at consistent zoom; store mapping id → image_path in CSV.

Preprocessing specifics: missing numeric values — median imputation; categorical — one-hot or target encoding; standardize using training set statistics.

(Report includes example images for low/high priced areas — see report visuals.) 

23116085_report

🧪 Models implemented

Tabular Baseline

Model: Gradient boosting regressor (preferred for tabular tasks).

Role: baseline benchmark.

Image-Only

Backbone: ResNet-18 (pretrained), embeddings → MLP.

Purpose: measure visual signal alone.

Multimodal Fusion

Strategy: early fusion (concatenate tabular features + image embeddings) → MLP.

Note: Simple concatenation was used; more advanced fusion (attention/late-fusion) is recommended for future work.
