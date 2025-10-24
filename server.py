from fastapi import FastAPI, WebSocket
import tensorflow as tf
import numpy as np
import json
import pickle
import os

# Load the trained model
model = tf.keras.models.load_model("toxicity_model.h5")

# Recreate the `TextVectorization` layer
MAX_WORDS = 200000
vectorizer = tf.keras.layers.TextVectorization(max_tokens=MAX_WORDS, output_sequence_length=1800, output_mode='int')

# Define Toxicity Labels
TOXICITY_LABELS = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]

# Load the saved vocabulary
if os.path.exists("vectorizer_vocab.pkl"):
    with open("vectorizer_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    if vocab:
        vectorizer.set_vocabulary(vocab)
        print("✅ Vocabulary loaded successfully!")
    else:
        print("🚨 Error: Loaded vocabulary is empty!")
        exit(1)
else:
    print("🚨 Error: vectorizer_vocab.pkl not found!")
    exit(1)

# Ensure Model is Compiled
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Store connected users
connected_users = set()

# Function to Predict Toxicity & Return Scores
def predict_toxicity(message):
    if not message.strip():
        return [0.0] * 6  

    input_text = tf.expand_dims(message, axis=0)
    input_text = vectorizer(input_text)

    prediction = model.predict(input_text)[0]
    toxicity_scores = {label : round(float(score),2) for label,score in zip(TOXICITY_LABELS, prediction)}
    return toxicity_scores  # Returns list of scores

# Start FastAPI
app = FastAPI()

@app.websocket("/chat")
async def chatroom(websocket: WebSocket):
    await websocket.accept()
    connected_users.add(websocket)

    try:
        while True:
            # Receive message from user
            data = await websocket.receive_text()
            message_data = json.loads(data)
            message = message_data["message"]
            user = message_data["user"]

            # Predict toxicity & get max score
            toxicity_scores = predict_toxicity(message)
            max_toxicity = max(toxicity_scores.values())

            # Format toxicity scores for display
            def format_toxicity_scores(toxicity_scores):
                return "\n".join(
                    f"{'🔴' if score >= 0.7 else '🟢'} {category}: {score:.2f}%"  
                    for category, score in toxicity_scores.items()
                )

            # If message is too toxic, block it
            if max_toxicity > 0.7:
                formatted_scores = format_toxicity_scores(toxicity_scores)
                await websocket.send_json({
                    "user": "assistant",
                    "message": "🚨 Message Blocked Due to High Toxicity!\n" + formatted_scores,
                    "toxicity_score": max_toxicity,  # Always include this
                    "toxicity_scores": formatted_scores  # Send formatted scores separately
                })
                continue  # Don't broadcast toxic messages

            # Broadcast Clean Messages + Toxicity Score to Everyone
            response_message = {
                "user": user,
                "message": message,
                "toxicity_score": max_toxicity  # Always include toxicity score
            }
            for user_socket in connected_users:
                await user_socket.send_json(response_message)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        connected_users.remove(websocket)  # Remove disconnected users


# Run FastAPI Server Locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

