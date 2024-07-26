import streamlit as st
import os
from crewai import Crew, Process
from agents import get_agents_and_tasks
from tools import send_mail

def generate_video(topic, openai_api_key, stabilityai_api_key, user_mail):
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
    if 'tmp' in result:
        result = result.split('/')[-1]
    send_mail(user_mail, result)
    return result

st.markdown("""
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
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="heading">
        YouTube Shorts Creator
    </div>
    <div class="subheading">
        Generate stunning short videos with simple text input
    </div>
    """, unsafe_allow_html=True)

st.text(" ")

topic = st.text_input('Enter title here')
openai_api_key = st.text_input('Enter your OpenAI API key here')
size = st.selectbox('Size',
                    ('512 X 512', '1024 X 1024', '9:16'))
stabilityai_api_key = None
if size=='9:16':
    stabilityai_api_key = st.text_input('Enter your stability ai API key here')
mail = st.text_input('Enter you email address')

if st.button('submit'):
    st.text(f"Video will be sent to {mail}")
    result = generate_video(topic, openai_api_key, stabilityai_api_key, mail)
    with open(result, 'rb') as video_file:
        video_data = video_file.read()
    st.video(video_data)

# Sidebar for Example Videos
st.sidebar.markdown("### Example Videos")
example_paths = os.listdir('results')
examples = [os.path.join('results', i) for i in example_paths]

# Display videos in a row
st.sidebar.markdown('<div class="example-video-container">', unsafe_allow_html=True)
for video_url in examples:
    # print(video_url)
    title = video_url.split('/')[1].split('.')[0]
    st.sidebar.text(f"input: {title}")
    st.sidebar.video(video_url, format="video/mp4", start_time=0)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

