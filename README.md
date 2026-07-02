# Text Toxicity Moderation

A complete, end-to-end machine learning project that trains a model to
detect toxic (abusive/insulting/harassing) text and moderates it — built for
Assignment #4.

**Pipeline:** raw text → cleaning → TF-IDF features → Logistic Regression →
threshold tuning → evaluation → `predict_toxicity()` function → Gradio demo
app → Hugging Face Spaces deployment.

> **v2 update:** retrained on a real-world public dataset (80,000-row
> sample of Civil Comments) instead of the original 100-row demo sample,
> with a tuned decision threshold to fix false positives on clearly polite
> text (e.g. "You are a wonderful person." was previously mis-flagged as
> toxic). See "Model & results" below.

## Project structure

```
.
├── text_toxicity_moderation.ipynb   # main notebook (run top-to-bottom)
├── app.py                           # Gradio demo app (local)
├── requirements.txt                 # project dependencies
├── data/
│   ├── prepare_real_dataset.py      # downloads & prepares the real Civil Comments dataset
│   ├── generate_sample_data.py      # regenerates the tiny offline fallback sample
│   ├── sample_toxicity_data.csv     # 100-row offline fallback sample (demo/pipeline-check only)
│   ├── raw/train.csv                # real dataset used for training (80,000 rows, committed)
│   └── README.md                    # dataset details + download instructions
├── models/
│   ├── toxicity_model.joblib        # trained pipeline (TF-IDF + Logistic Regression)
│   └── metadata.joblib              # model choice, threshold, and evaluation metrics
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

# 2. (Optional) regenerate the real dataset from Hugging Face (already committed at data/raw/train.csv)
pip install datasets
python data/prepare_real_dataset.py

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

Trained on **80,000 real comments** sampled from
[`google/civil_comments`](https://huggingface.co/datasets/google/civil_comments)
(the dataset behind the Jigsaw "Unintended Bias in Toxicity Classification"
competition), downloaded with no authentication required and committed to
this repo at `data/raw/train.csv`. Civil Comments was used instead of the
Kaggle-hosted Jigsaw Toxic Comment Classification CSV because the latter
requires a personal Kaggle account/API token; instructions to swap it in
anyway are in `data/README.md`. A tiny 100-row hand-written sample
(`data/sample_toxicity_data.csv`) remains as a zero-setup fallback only —
it is **not** used for the model shipped in this repo, since it's too small
to generalize (that's what originally caused false positives — see below).

## Model & results

**TF-IDF + Logistic Regression** (`max_features=40000`, unigrams+bigrams,
`min_df=2`, mild `class_weight={0: 1, 1: 2}`), trained on 63,769 rows,
evaluated on a held-out 15,943-row test set, with a tuned decision
**threshold of 0.30** (instead of the naive 0.5 cutoff) chosen specifically
to reduce false positives on clean/polite text:

| metric | value |
|---|---|
| Accuracy | 0.935 |
| Precision (toxic) | 0.595 |
| Recall (toxic) | 0.514 |
| F1 (toxic) | 0.552 |

Full confusion matrix, precision/recall-vs-threshold sweep, and
false-positive/false-negative examples are in notebook Section 6.
`models/metadata.joblib` stores these numbers programmatically.

**Required sentence verification** (notebook Section 7, asserted in-notebook):

| text | expected | predicted | toxic score |
|---|---|---|---|
| "Hello" | Non-toxic | Non-toxic | 0.047 |
| "Thank you" | Non-toxic | Non-toxic | 0.021 |
| "You are a wonderful person." | Non-toxic | Non-toxic | 0.220 |
| "You are stupid." | Toxic | Toxic | 1.000 |
| "I hate you." | Toxic | Toxic | 0.362 |
| "You are useless." | Toxic | Toxic | 0.789 |

> This is a lightweight classic-ML baseline, not a state-of-the-art
> transformer — precision/recall on ambiguous or sarcastic text will still
> be imperfect. See notebook Section 9 for how to source more data and
> Section 8 for a path to a stronger (multilingual) model.

## Gradio demo

`app.py` loads `models/toxicity_model.joblib` + `models/metadata.joblib`
(for the tuned threshold) and provides a simple UI: enter text → click
**Analyze** → see the Toxic/Non-toxic label, a confidence score, and
friendly moderation advice.

## Deploying to Hugging Face Spaces

The `huggingface_space/` folder is a ready-to-upload package (already
updated with the retrained model). See `huggingface_space/README.md` (and
notebook Section 11) for full steps:

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

Educational assignment project. Trained on a real public dataset but at a
small-to-moderate scale — not a production-grade content moderation system
without further validation, bias auditing, and a larger training set.
