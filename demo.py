import streamlit as st
import os
from crewai import Crew, Process
from agents import get_agents_and_tasks
from tools import send_mail

def generate_video(topic, openai_api_key, stabilityai_api_key, user_email):
    os.environ['OPENAI_API_KEY'] = openai_api_key
    grow_api_key = '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'
    if stabilityai_api_key is not None:
        os.environ['STABILITY_AI_API_KEY'] = stabilityai_api_key
        model = 'Stability AI'
    else:
        model = 'Dalle-2'
    agents, tasks = get_agents_and_tasks(grow_api_key)

    crew = Crew(
        agents=agents, 
        tasks=tasks,
        process=Process.sequential,
        memory=True,
        verbose=2
    )
    result = crew.kickoff(inputs={'topic': topic, 'model' : model})
    send_mail(user_email, result)
    return result

# Custom CSS for styling
st.markdown(
    """
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
        .example-video-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-start;
        }
        .example-video {
            margin: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main content area
st.markdown(
    """
    <div class="heading">
        YouTube Shorts Creator
    </div>
    <div class="subheading">
        Generate stunning short videos with simple prompt
    </div>
    """,
    unsafe_allow_html=True,
)

# Space
st.text(" ")

# Main content for Input Details
st.markdown("## Input Details")
topic = st.text_input("Prompt")
openai_api_key = st.text_input("OpenAI API key")
size = st.selectbox('Size',
                    ('512 X 512', '1024 X 1024', '9:16'))
stabilityai_api_key = None
if size=='9:16':
    st.text_input('Stability Ai API key')
user_email = st.text_input("Email address", placeholder='yourname@gmail.com')

if st.button("Mail Me!"):
    st.text(f"Video will be sent to {user_email}")
    result = generate_video(topic, openai_api_key, stabilityai_api_key, user_email)
    #st.success(result)
    # In a real scenario, you would send the generated video to the user's email here

# Sidebar for Example Videos
st.sidebar.markdown("### Example Videos")
example_paths = os.listdir('results')
examples = [os.path.join('results', i) for i in example_paths]

# Display videos in a row
st.sidebar.markdown('<div class="example-video-container">', unsafe_allow_html=True)
for video_url in examples:
    title = video_url.split('\\')[1].split('.')[0]
    st.sidebar.text(f"input: {title}")
    st.sidebar.video(video_url, format="video/mp4", start_time=0)
st.sidebar.markdown("</div>", unsafe_allow_html=True)
