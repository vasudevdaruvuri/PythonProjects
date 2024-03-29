import gradio as gr

def word_count(text):
    return len(text.split())

input_text = gr.Textbox(lines=5, label="Enter text")
#output_text = gr.outputs.Textbox(label="Total words")

gr.Interface(fn=word_count, inputs=input_text, outputs="text").launch()