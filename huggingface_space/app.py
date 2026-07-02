"""
app.py
------
Gradio demo app for the Text Toxicity Moderation model.

Run locally:
    pip install -r requirements.txt
    python app.py

Then open the printed local URL (e.g. http://127.0.0.1:7860) in a browser.

This app loads the trained pipeline saved by text_toxicity_moderation.ipynb
(models/toxicity_model.joblib) and exposes a simple "enter text -> analyze"
UI so a trainer/reviewer can test the model without writing any code.
"""

import os
import re

import gradio as gr
import joblib

MODEL_PATH = os.path.join("models", "toxicity_model.joblib")

# ---------------------------------------------------------------------------
# Text cleaning — must match the preprocessing used during training
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
SPECIAL_CHARS_RE = re.compile(r"[^a-z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/special characters, and normalize whitespace."""
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = SPECIAL_CHARS_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


print(f"Loading model from {MODEL_PATH} ...")
model = joblib.load(MODEL_PATH)
print("Model loaded.")


def predict_toxicity(text: str, threshold: float = 0.5) -> dict:
    """Predict whether a piece of text is toxic. See notebook Section 7 for details."""
    if not text or not str(text).strip():
        return {
            "label": "Non-toxic",
            "confidence": 1.0,
            "toxic_score": 0.0,
            "explanation": "Empty input — nothing to moderate.",
        }

    cleaned = clean_text(text)
    proba = model.predict_proba([cleaned])[0]
    toxic_score = float(proba[1])
    is_toxic = toxic_score >= threshold
    label = "Toxic" if is_toxic else "Non-toxic"
    confidence = toxic_score if is_toxic else 1 - toxic_score

    if is_toxic and toxic_score >= 0.85:
        explanation = (
            "This message contains strong indicators of insulting/abusive "
            "language and should likely be blocked or sent for review."
        )
    elif is_toxic:
        explanation = (
            "This message shows moderate signs of toxic language "
            "(e.g. insults or hostile tone) and should be flagged for review."
        )
    else:
        explanation = (
            "No strong indicators of toxic language were detected; "
            "this message looks safe to publish."
        )

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "toxic_score": round(toxic_score, 4),
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def analyze(text):
    result = predict_toxicity(text)

    if result["label"] == "Toxic":
        verdict = f"🚫 **TOXIC** (toxicity score: {result['toxic_score']:.0%})"
    else:
        verdict = f"✅ **Non-toxic** (toxicity score: {result['toxic_score']:.0%})"

    confidence_text = f"Model confidence: {result['confidence']:.0%}"
    advice = result["explanation"]

    return verdict, confidence_text, advice


EXAMPLES = [
    "Thank you so much for your help, this really solved my problem!",
    "You are such an idiot, nobody wants to hear your opinion.",
    "Could you please clarify the refund policy for this order?",
    "Shut up already, your comments are garbage and so are you.",
    "I really appreciate the quick response from the support team.",
    "Get lost, you pathetic loser, nobody likes you here.",
]

with gr.Blocks(title="Text Toxicity Moderation") as demo:
    gr.Markdown(
        """
        # 🛡️ Text Toxicity Moderation Demo

        Enter any text message (comment, chat message, review, etc.) below
        and click **Analyze** to check whether it would likely be flagged
        as toxic by an automated moderation filter.

        This demo uses a TF-IDF + classic ML classifier trained in
        `text_toxicity_moderation.ipynb`. See the notebook for full details
        on the dataset, training, and evaluation.
        """
    )

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Enter text to analyze",
                placeholder="Type or paste a comment / message here...",
                lines=4,
            )
            analyze_btn = gr.Button("🔍 Analyze", variant="primary")
            gr.Examples(examples=EXAMPLES, inputs=text_input, label="Try an example")

        with gr.Column():
            verdict_output = gr.Markdown(label="Result")
            confidence_output = gr.Textbox(label="Confidence", interactive=False)
            advice_output = gr.Textbox(label="Moderation Advice", interactive=False, lines=3)

    analyze_btn.click(
        fn=analyze,
        inputs=text_input,
        outputs=[verdict_output, confidence_output, advice_output],
    )
    text_input.submit(
        fn=analyze,
        inputs=text_input,
        outputs=[verdict_output, confidence_output, advice_output],
    )

    gr.Markdown(
        """
        ---
        ⚠️ **Disclaimer:** This is an educational demo trained on a small
        sample dataset. It is not intended for production moderation
        decisions without further validation on a larger, representative
        dataset.
        """
    )

if __name__ == "__main__":
    demo.launch()
