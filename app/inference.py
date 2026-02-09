# app/inference.py
import pickle
import numpy as np
import tensorflow as tf

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

MODEL_PATH = "artifacts/toxicity_model.h5"
VOCAB_PATH = "artifacts/vectorizer_vocab.pkl"

MAX_WORDS = 200000
SEQ_LEN = 1800

# 1) Load model
model = tf.keras.models.load_model(MODEL_PATH)

# 2) Recreate vectorizer exactly like training
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_WORDS,
    output_sequence_length=SEQ_LEN,
    output_mode="int",
)

# 3) Load vocab and set it
with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)
vectorizer.set_vocabulary(vocab)

def predict_scores(text: str) -> dict:
    # vectorizer expects a batch (list/array)
    x = tf.constant([text])
    x = vectorizer(x)

    y = model.predict(x, verbose=0)[0]  # shape: (6,)
    y = np.array(y, dtype=float)

    return {label: float(score) for label, score in zip(LABELS, y)}
