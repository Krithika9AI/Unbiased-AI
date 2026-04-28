# Unbiased-AI
AI Bias Detection Tool for Google Solution Challenge 2026

# ⚖️ Unbiased AI

### Ensuring Fairness and Detecting Bias in Automated Decisions

**Google Solution Challenge 2026**

## Problem
AI systems are making important decisions about jobs, bank loans, and medical care. But if they learn from old unfair data, they repeat and amplify discrimination.

## Solution
**Unbiased AI** is a tool that helps organizations detect hidden bias in their AI models before deploying them.

### Features
- Upload dataset or use example (Adult Income Dataset)
- Train a simple AI model
- Measure bias using fairness metrics (Disparate Impact, Approval Rate by Group)
- Show clear warnings (High / Moderate / Low Bias)
- Visual charts to understand bias easily

## Tech Stack
- **Frontend**: Designed using Google Stitch
- **Backend**: Streamlit + Scikit-learn + Plotly
- **Fairness Analysis**: Custom group metrics

## How to Run Locally

```bash
pip install streamlit pandas scikit-learn plotly numpy
streamlit run app.py
