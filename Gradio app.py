import gradio as gr
import spaces
import torch

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

zero = torch.tensor([0.0], device=device)

@spaces.GPU
def greet(n):
    value = zero + float(n)
    return f"Hello {value.item()} on {zero.device}"


demo = gr.Interface(fn=greet, inputs=gr.Number(), outputs=gr.Text())
demo.launch()