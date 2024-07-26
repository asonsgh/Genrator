import gradio as gr
import os

from crewai import Crew, Process
from agents import get_agents_and_tasks


def generate_video(topic, openai_api_key, stabilityai_api_key):
    # os.environ['STABILITY_AI_API_KEY'] = stability_ai_api_key
    os.environ['OPENAI_API_KEY'] = openai_api_key
    grow_api_key = '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'
    agents, tasks = get_agents_and_tasks(grow_api_key)

    crew = Crew(
    agents = agents, 
    tasks = tasks,
    process = Process.sequential,
    memory=True,
    verbose=2
    )
    result = crew.kickoff(inputs={'topic': topic})
    return result

intro = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shorts Generator Page</title>
    <style>
        .heading {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
        }
        .subheading {
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            margin-top: 10px;
        }
        .additional-text {
            font-size: 16px;
            text-align: center;
            margin-top: 20px;
            white-space: pre-line; /* Preserve line breaks */
        }
    </style>
</head>
<body>
    <div class="heading">
        YouTube Shorts Creator
    </div>
    <div class="subheading">
        Generate stunning short videos with simple text input
    </div>
</body>
</html>
"""

with gr.Blocks(title='ShortsIn') as app:

  gr.HTML(intro)

  with gr.Row():
    with gr.Column():
      inp = gr.Textbox(label='Enter title here')
      api = gr.Textbox(label='Enter your openai API key here')
      btn = gr.Button('Generate', size='lg')

    with gr.Column():
      out = gr.Video(label='')
      

  btn.click(fn=generate_video, inputs=[inp, api], outputs=out)

app.launch()
