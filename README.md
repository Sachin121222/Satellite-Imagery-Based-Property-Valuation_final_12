# 🛰️ Satellite Imagery–Based Property Valuation

**CDC × Yhills Open Projects (2025–26)**  
**Domain:** Data Science  
**Tech Stack:** Python, PyTorch, Scikit-learn  

A multimodal machine learning project that predicts residential property prices by combining **structured housing data** with **satellite imagery**. The project studies whether visual neighborhood context can complement traditional tabular features in real estate valuation.

---

## 📌 Table of Contents
- Overview  
- Problem Statement  
- Methodology  
- Project Structure  
- Models  
- Results  
- Model Explainability  
- Setup  
- Usage  
- Limitations  

---

## 🔍 Overview

Most automated property valuation systems rely on structured attributes such as living area, number of rooms, and location coordinates. However, neighborhood characteristics like green spaces, water proximity, and urban density are often not explicitly captured.

This project integrates:
- **Tabular property features** describing internal characteristics
- **Satellite imagery** capturing external neighborhood context  

The goal is to evaluate the impact of satellite imagery on property price prediction when used alongside traditional features.

---

## ❓ Problem Statement

Traditional housing datasets lack sufficient environmental and neighborhood information, which can lead to inaccurate valuations.

**Research Question:**  
Can satellite imagery improve residential property price prediction when combined with structured housing data?

---

## ⚙️ Methodology

The project follows a multimodal machine learning pipeline:

