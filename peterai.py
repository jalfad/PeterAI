import torch
import gradio as gr
from transformers import pipeline
chatbox = pipeline("text-generation", model="gpt2")

history = """
You are PeterAI, a smart AI who talks like Spider-Man. Always answer in Spider-Man style. Use action and humor.

"""
def ai_response(message):
    global history
    history += f"\nUSER: {message}\nPeterAI:"

    result = chatbox(
        history,
        max_length=100,
        do_sample = True,
        temperature=0.8,
        top_k=50,
        top_p=0.95
    )
    reply = result[0]['generated_text'].split("PeterAI:")[-1].strip().split('\n')[0]
    history += f"{reply}"
    return reply

demo = gr.Interface(
    fn=ai_response,
    inputs="text",
    outputs="text",
    title="Peter AI - Spider-Man Chat",
    description="Type your message and PeterAI will reply in Spider-Man style!"
)

demo.launch()
