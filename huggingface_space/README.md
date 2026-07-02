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

- **Model:** TF-IDF (word 1-2 grams) + Logistic Regression / Linear SVM
  (whichever scored higher F1 during training), scikit-learn `Pipeline`.
- **Training data:** see the parent project's `text_toxicity_moderation.ipynb`
  notebook and `data/README.md` for dataset details (sample dataset shipped
  in-repo, with instructions to swap in the full Jigsaw Toxic Comment
  Classification dataset for production use).
- **Output:** Toxic / Non-toxic label, a confidence score, and a friendly
  moderation-advice message.

## Try these examples

| Text | Expected |
|---|---|
| "Thank you so much for your help, this really solved my problem!" | Non-toxic |
| "You are such an idiot, nobody wants to hear your opinion." | Toxic |
| "Could you please clarify the refund policy for this order?" | Non-toxic |
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

This is an educational assignment demo trained on a small sample dataset. It
is not intended for production moderation decisions without retraining on a
larger, representative, and bias-audited dataset (see the notebook's
"Multilingual Discussion" and "Real-world Data Sources" sections).
