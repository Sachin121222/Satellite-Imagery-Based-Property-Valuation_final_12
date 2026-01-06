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

