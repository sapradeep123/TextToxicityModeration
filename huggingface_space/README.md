---
title: Text Toxicity Moderation
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🛡️ Text Toxicity Moderation — Live Demo

Type any comment/message into the box and click **Analyze** to see whether
an automated moderation classifier would flag it as toxic.

- **Model:** TF-IDF (word 1-2 grams) + Logistic Regression, scikit-learn
  `Pipeline`, with a tuned decision threshold (0.30, stored in
  `models/metadata.joblib` and loaded automatically by `app.py`).
- **Training data:** 80,000 real comments sampled from
  [`google/civil_comments`](https://huggingface.co/datasets/google/civil_comments)
  (the dataset behind the Jigsaw "Unintended Bias in Toxicity
  Classification" competition). See the parent project's
  `text_toxicity_moderation.ipynb` notebook and `data/README.md` for full
  dataset details.
- **Output:** Toxic / Non-toxic label, a confidence score, and a friendly
  moderation-advice message.
- **Test-set metrics:** accuracy 0.935, precision 0.595, recall 0.514,
  F1 0.552 (toxic class; see the notebook for the full confusion matrix and
  threshold sweep).

## Try these examples

| Text | Expected |
|---|---|
| "Hello" | Non-toxic |
| "Thank you" | Non-toxic |
| "You are a wonderful person." | Non-toxic |
| "You are stupid." | Toxic |
| "I hate you." | Toxic |
| "You are useless." | Toxic |
| "Thank you so much for your help, this really solved my problem!" | Non-toxic |
| "Shut up already, your comments are garbage and so are you." | Toxic |

## Files in this Space

- `app.py` — Gradio app (loads `models/toxicity_model.joblib`).
- `models/toxicity_model.joblib` — trained scikit-learn pipeline (TF-IDF + classifier).
- `models/metadata.joblib` — training metadata (model choice, metrics, dataset source).
- `requirements.txt` — minimal runtime dependencies for this Space.

## Uploading this folder to Hugging Face Spaces

1. Create a free account at [huggingface.co](https://huggingface.co).
2. **New Space** → pick a name → SDK = **Gradio** → visibility = **Public**
   → **Create Space**.
3. Push the contents of this folder to the Space repo:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/text-toxicity-moderation
   cp -r huggingface_space/* text-toxicity-moderation/
   cd text-toxicity-moderation
   git add .
   git commit -m "Add toxicity moderation Gradio app"
   git push
   ```
4. The Space builds automatically and gives you a public URL:
   `https://huggingface.co/spaces/<your-username>/text-toxicity-moderation`

## Disclaimer

This is an educational assignment demo trained on a real but moderately
sized public dataset (80,000 rows). It is not intended for production
moderation decisions without retraining on a larger, representative, and
bias-audited dataset (see the notebook's "Multilingual Discussion" and
"Real-world Data Sources" sections).
