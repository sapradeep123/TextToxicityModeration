# Text Toxicity Moderation

A complete, end-to-end machine learning project that trains a model to
detect toxic (abusive/insulting/harassing) text and moderates it — built for
Assignment #4.

**Pipeline:** raw text → cleaning → TF-IDF features → Logistic Regression /
Linear SVM → evaluation → `predict_toxicity()` function → Gradio demo app →
Hugging Face Spaces deployment.

## Project structure

```
.
├── text_toxicity_moderation.ipynb   # main notebook (run top-to-bottom)
├── app.py                           # Gradio demo app (local)
├── requirements.txt                 # project dependencies
├── data/
│   ├── generate_sample_data.py      # regenerates the offline sample dataset
│   ├── sample_toxicity_data.csv     # 100-row offline sample (50 toxic / 50 clean)
│   └── README.md                    # dataset details + real-dataset download instructions
├── models/
│   ├── toxicity_model.joblib        # trained pipeline (TF-IDF + classifier)
│   └── metadata.joblib              # model choice + evaluation metrics
└── huggingface_space/                # self-contained package ready to deploy to HF Spaces
    ├── app.py
    ├── requirements.txt
    ├── README.md                     # includes HF Spaces YAML config header
    └── models/
        ├── toxicity_model.joblib
        └── metadata.joblib
```

## What is text toxicity moderation?

Automatically detecting whether user-generated text (comments, chat
messages, reviews, support tickets, forum posts) is toxic — rude, insulting,
harassing, or threatening — versus civil/on-topic. It's used to keep online
communities safe, protect brand/legal risk, speed up human moderation
workflows, and triage hostile customer support interactions. See Section 1
of the notebook for the full discussion.

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) regenerate the sample dataset
python data/generate_sample_data.py

# 3. Run the notebook top-to-bottom (Jupyter, or via nbconvert)
jupyter notebook text_toxicity_moderation.ipynb
# or, non-interactively:
jupyter nbconvert --to notebook --execute --inplace text_toxicity_moderation.ipynb

# 4. Launch the Gradio demo (uses the model saved by the notebook)
python app.py
```

Open the local URL printed by `app.py` (e.g. `http://127.0.0.1:7860`) in a
browser, type text into the box, and click **Analyze**.

## Dataset

Ships with a small, hand-curated **offline sample dataset**
(`data/sample_toxicity_data.csv`, 100 rows) so the notebook runs with zero
setup. For a real, production-quality model, download the
[Jigsaw Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)
dataset (or Civil Comments) and drop it at `data/raw/train.csv` — the
notebook automatically picks it up. Full instructions: `data/README.md`.

## Model & results

Two classic, fast, GPU-free models are trained and compared on TF-IDF
features (word unigrams + bigrams):

- **Option A:** TF-IDF + Logistic Regression (calibrated probabilities out of the box)
- **Option B:** TF-IDF + Linear SVM (via `CalibratedClassifierCV` for probabilities)

The notebook automatically selects whichever scores higher F1 on the held-out
test set, reports accuracy/precision/recall/F1, a confusion matrix, and
inspects false-positive/false-negative examples. Exact numbers depend on
which dataset you train on (sample vs. real) — see the notebook output for
the run's actual metrics (`models/metadata.joblib` stores them
programmatically).

> **Note:** with only the 100-row sample dataset, metrics and confidence
> scores are modest — this is expected for a tiny demo dataset and is
> discussed in the notebook. Swap in the real Jigsaw dataset (see
> `data/README.md`) for a model you'd actually trust in production.

## Gradio demo

`app.py` loads `models/toxicity_model.joblib` and provides a simple UI:
enter text → click **Analyze** → see the Toxic/Non-toxic label, a confidence
score, and friendly moderation advice.

## Deploying to Hugging Face Spaces

The `huggingface_space/` folder is a ready-to-upload package. See
`huggingface_space/README.md` (and notebook Section 11) for full steps:

1. Create a Space on [huggingface.co](https://huggingface.co) with SDK = Gradio.
2. Push the contents of `huggingface_space/` to the Space's git repo.
3. Get a public URL like `https://huggingface.co/spaces/<username>/text-toxicity-moderation`
   — share it with your trainer, no install required.

## Multilingual support & real-world data sourcing

Covered in depth in notebook Sections 8 and 9: translation vs. multilingual
transformer approaches, per-language data requirements, where to responsibly
source more toxic/non-toxic examples, and privacy/consent/bias
considerations.

## Submission checklist

- [ ] `text_toxicity_moderation.ipynb` — run top-to-bottom with outputs saved
- [ ] GitHub repo pushed: `git@github.com:sapradeep123/TextToxicityModeration.git`
- [ ] Hugging Face Space deployed and public link obtained
- [ ] This `README.md`

## Disclaimer

Educational assignment project. The bundled sample dataset and trained model
are for demonstration only — not a production-grade content moderation
system without further validation, bias auditing, and training on a larger
real-world dataset.
