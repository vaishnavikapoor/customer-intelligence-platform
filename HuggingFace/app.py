import gradio as gr
from rag_engine import answer_with_sources

def rag_interface(question, k):
    """
    Calls the RAG engine to get an answer and sources for a given question.
    """
    result = answer_with_sources(question, k)
    answer_text = result["answer"]
    sources_text = "\n".join(result["sources"]) if result.get("sources") else "No relevant complaint records found."
    return answer_text, sources_text

with gr.Blocks(title="Customer Intelligence Platform") as demo:
    # Header
    gr.Markdown("## Customer Intelligence Platform")
    gr.Markdown(
        "This system retrieves relevant customer complaint records and generates "
        "grounded, source-backed answers. Out-of-scope questions will not be answered."
    )

    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(
                label="Enter your question",
                placeholder="Why are customers complaining about refunds?",
                lines=2
            )
            k = gr.Slider(
                minimum=2, maximum=10, value=4, step=1,
                label="Context breadth (Top K retrieved chunks)",
                info="Higher values retrieve more context but may add noise and slow responses."
            )
            btn = gr.Button("Ask")

        with gr.Column(scale=3):
            answer = gr.Textbox(label="Answer", lines=10)
            sources = gr.Textbox(label="Sources", lines=10)

    btn.click(fn=rag_interface, inputs=[question, k], outputs=[answer, sources])

demo.launch()
