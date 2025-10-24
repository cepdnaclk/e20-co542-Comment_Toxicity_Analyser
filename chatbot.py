import gradio as gr
import websockets
import asyncio
import json

# WebSocket URL (FastAPI Backend)
WEBSOCKET_URL = "ws://localhost:8000/chat"

# Store username globally
username = ""

# Function to Connect to WebSocket & Send Messages
async def send_message(message, history, user):
    if not user.strip():
        return history + [{"role": "assistant", "content": "⚠️ Please enter a username before sending messages."}]

    async with websockets.connect(WEBSOCKET_URL) as websocket:
        user_message = {"user": user, "message": message}
        await websocket.send(json.dumps(user_message))

        response = await websocket.recv()
        response_data = json.loads(response)

        # Show Sent Message
        history.append({"role": "user", "content": f"👤 {user}: {message}"})

        # If the message was blocked, display formatted toxicity scores
        if "toxicity_scores" in response_data:
            history.append({"role": "assistant", "content": response_data["message"]})  # Show blocked message alert
            return history  # Do not continue, stop here

        # Show response with toxicity score
        history.append({
            "role": "assistant",
            "content": f"🤖 Assistant: {response_data['message']} (Toxicity: {response_data['toxicity_score']:.2f})"
        })

        return history
 

# Gradio Chat UI
with gr.Blocks() as demo:
    username_input = gr.Textbox(label="Enter Your Name", placeholder="Type your name here...")
    chat_interface = gr.Chatbot([], type="messages", label="Toxicity Detection Chatroom")
    msg_input = gr.Textbox(label="Type your message here:")
    send_btn = gr.Button("Send")

    # Send Messages & Show Toxicity Stats
    send_btn.click(fn=send_message, inputs=[msg_input, chat_interface, username_input], outputs=chat_interface)

# Launch Gradio UI
demo.launch()
