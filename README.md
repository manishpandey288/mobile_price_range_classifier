# Mobile Price Range Classification

A Machine Learning web application that predicts the **price category** of a mobile phone based on its hardware, display, and connectivity specifications. Built with **Scikit-Learn** and deployed using **Streamlit**.

---

## Overview

Determining the right price segment for a smartphone depends on multiple hardware factors such as RAM, battery capacity, screen resolution, internal storage, and processor specifications. This project trains and compares multiple classification algorithms on mobile specification data and serves the best-performing model (**Logistic Regression – 95.83% accuracy**) through an interactive web interface.

### Price Categories
| Class | Price Segment | Estimated Bracket |
| :--- | :--- | :--- |
| **0** | Low Cost (Budget) | ₹5,000 – ₹10,000 |
| **1** | Medium Cost (Mid-Range) | ₹10,000 – ₹20,000 |
| **2** | High Cost (Upper Mid-Range) | ₹20,000 – ₹35,000 |
| **3** | Very High Cost (Flagship) | ₹35,000+ |

---

## 📊 Dataset & Features

The dataset (`mobile_price_range_data.csv`) contains **2,000 samples** and **20 input features** along with the target label `price_range`:

- **Performance & Memory:** `ram`, `battery_power`, `int_memory`, `clock_speed`, `n_cores`
- **Camera:** `pc` (Primary Camera MP), `fc` (Front Camera MP)
- **Display & Physical Specs:** `px_height`, `px_width`, `sc_h`, `sc_w`, `mobile_wt`, `m_dep`, `talk_time`
- **Connectivity & Extras:** `four_g`, `three_g`, `wifi`, `blue`, `dual_sim`, `touch_screen`

---

## Model Comparison

Multiple classifiers were evaluated using a **70:30 train-test split** with `StandardScaler` feature scaling:

| Model | Test Accuracy |
| :--- | :---: |
| **Logistic Regression** | **95.83%** |
| SVM (Linear Kernel, `C=0.01`) | 90.50% |
| SVM (RBF Kernel, `C=1`) | 87.00% |
| Decision Tree (Entropy) | 82.83% |
| KNN (`n_neighbors=15`) | 56.17% |

> **Logistic Regression** achieved the highest test accuracy (**95.83%**) and is used in the web application for real-time predictions.

---

## Tech Stack

- **Language:** Python 3
- **Data Processing & ML:** Pandas, NumPy, Scikit-Learn
- **Visualization (Notebook):** Matplotlib, Seaborn
- **Web Framework:** Streamlit

---

## Project Structure

```text
├── app.py                              # Streamlit web application
├── Mobile_price_classification.ipynb   # Exploratory analysis & model training notebook
├── mobile_price_range_data.csv         # Dataset (2000 rows, 21 columns)
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
