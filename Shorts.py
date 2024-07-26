import gradio as gr
import os
import re
import tempfile
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from gtts import gTTS
from pydub import AudioSegment
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, ImageClip
from crewai import Crew, Process, Agent, Task
from langchain_groq import ChatGroq
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.pydantic_v1 import BaseModel, Field
from openai import OpenAI
import cv2
import numpy as np

def split_text_into_chunks(text, chunk_size):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def add_text_to_video(input_video, text, duration=1, fontsize=40, fontcolor=(255, 255, 255),
                      outline_thickness=2, outline_color=(0, 0, 0), delay_between_chunks=0.3,
                      font_path='Montserrat-Bold.ttf'):
    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_video = temp_output_file.name

    chunks = split_text_into_chunks(text, 3)  # Adjust chunk size as needed

    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    chunk_duration_frames = duration * fps
    delay_frames = int(delay_between_chunks * fps)

    font = ImageFont.truetype(font_path, fontsize)

    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        chunk_index = current_frame // (chunk_duration_frames + delay_frames)

        if current_frame % (chunk_duration_frames + delay_frames) < chunk_duration_frames and chunk_index < len(chunks):
            chunk = chunks[chunk_index]
            text_bbox = draw.textbbox((0, 0), chunk, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            text_x = (width - text_width) // 2
            text_y = height - 400  # Position text at the bottom

            if text_width > width:
                words = chunk.split()
                half = len(words) // 2
                line1 =''.join(words[:half])
                line2 =''.join(words[half:])

                text_size_line1 = draw.textsize(line1, font=font)
                text_size_line2 = draw.textsize(line2, font=font)
                text_x_line1 = (width - text_size_line1[0]) // 2
                text_x_line2 = (width - text_size_line2[0]) // 2
                text_y = height - 250 - text_size_line1[1]  # Adjust vertical position for two lines

                for dx in range(-outline_thickness, outline_thickness + 1):
                    for dy in range(-outline_thickness, outline_thickness + 1):
                        if dx!= 0 or dy!= 0:
                            draw.text((text_x_line1 + dx, text_y + dy), line1, font=font, fill=outline_color)
                            draw.text((text_x_line2 + dx, text_y + text_size_line1[1] + dy), line2, font=font, fill=outline_color)
                
                draw.text((text_x_line1, text_y), line1, font=font, fill=fontcolor)
                draw.text((text_x_line2, text_y + text_size_line1[1]), line2, font=font, fill=fontcolor)

            else:
                for dx in range(-outline_thickness, outline_thickness + 1):
                    for dy in range(-outline_thickness, outline_thickness + 1):
                        if dx!= 0 or dy!= 0:
                            draw.text((text_x + dx, text_y + dy), chunk, font=font, fill
                                                      draw.text((text_x, text_y), chunk, font=font, fill=fontcolor)

            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        out.write(frame)
        current_frame += 1

        # Ensure loop breaks after processing all frames
        if current_frame >= frame_count:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return output_video

def apply_zoom_in_effect(clip, zoom_factor=1.2):
    width, height = clip.size
    duration = clip.duration

    def zoom_in_effect(get_frame, t):
        frame = get_frame(t)
        zoom = 1 + (zoom_factor - 1) * (t / duration)
        new_width, new_height = int(width * zoom), int(height * zoom)
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        x_start = (new_width - width) // 2
        y_start = (new_height - height) // 2
        cropped_frame = resized_frame[y_start:y_start + height, x_start:x_start + width]
        
        return cropped_frame

    return clip.fl(zoom_in_effect, apply_to=['mask'])

def create_video_from_images_and_audio(images_dir, speeches_dir, final_video_filename):
    """Creates video using images and audios.
    Args:
    images_dir: path to images folder
    speeches_dir: path to speeches folder
    final_video_filename: the topic name which will be used as final video file name"""
    client = Groq(api_key='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')
    images_paths = sorted([os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.endswith('.png') or img.endswith('.jpg')])
    audio_paths = sorted([os.path.join(speeches_dir, speech) for speech in os.listdir(speeches_dir) if speech.endswith('.mp3')])
    clips = []
    temp_files = []
    
    for i in range(min(len(images_paths), len(audio_paths))):
        img_clip = ImageClip(os.path.join(images_dir, images_paths[i]))
        audioclip = AudioFileClip(os.path.join(speeches_dir, audio_paths[i]))
        videoclip = img_clip.set_duration(audioclip.duration)
        zoomed_clip = apply_zoom_in_effect(videoclip, 1.3)
        
        with open(os.path.join(speeches_dir, audio_paths[i]), "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(audio_paths[i], file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
            caption = transcription.text
        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        zoomed_clip.write_videofile(temp_video_path, codec='libx264', fps=24)
        temp_files.append(temp_video_path)
        
        final_video_path = add_text_to_video(temp_video_path, caption, duration=1, fontsize=60)
        temp_files.append(final_video_path)
        
        final_clip = VideoFileClip(final_video_path)
        final_clip = final_clip.set_audio(audioclip)

        print(f'create small video {i}')
        clips.append(final_clip)
    
    final_clip = concatenate_videoclips(clips)
    if not final_video_filename.endswith('.mp4'):
        final_video_filename = final_video_filename + '.mp4'
    final_clip.write_videofile(final_video_filename, codec='libx264', fps=24)
    
    # Close all video files properly
    for clip in clips:
        clip.close()
        
    # Remove all temporary files
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except Exception as e:
            print(f"Error removing file {temp_file}: {e}")
    
    return final_video_filename

class WikiInputs(BaseModel):
    """Inputs to the wikipedia tool."""
    query: str = Field(description="query to look up in Wikipedia, should be 3 or less words")

api_wrapper = WikipediaAPIWrapper(top_k_results=2)

wiki_tool = WikipediaQueryRun(
    name="wiki-tool",
    description="{query:'input here'}",
    args_schema=WikiInputs,
    api_wrapper=api_wrapper,
    return_direct=True,
)

def process_script(script):
    """Used to process the script into dictionary format"""
    dict = {}
    text_for_image_generation = re.findall(r'<image>(.*?)</?image>', script, re.DOTALL)
    text_for_speech_generation = re.findall(r'<narration>(.*?)</?narration>', script, re.DOT
         dict['text_for_image_generation'] = text_for_image_generation
    dict['text_for_speech_generation'] = text_for_speech_generation
    return dict

def generate_speech(text, lang='en', speed=1.15, num=0):
    """
    Generates speech for the given script using gTTS and adjusts the speed.
    """
    temp_speech_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    temp_speech_path = temp_speech_file.name

    tts = gTTS(text=text, lang=lang)
    tts.save(temp_speech_path)

    sound = AudioSegment.from_file(temp_speech_path)
    if speed!= 1.0:
        sound_with_altered_speed = sound._spawn(sound.raw_data, overrides={
            "frame_rate": int(sound.frame_rate * speed)
        }).set_frame_rate(sound.frame_rate)
        sound_with_altered_speed.export(temp_speech_path, format="mp3")
    else:
        sound.export(temp_speech_path, format="mp3")

    temp_speech_file.close()
    return temp_speech_path

def image_generator(script, model):
    """Generates images for the given script.
    Saves it to a temporary directory and returns the path.
    Args:
    script: a complete script containing narrations and image descriptions.
    model: image generation model used to generate images, can be 'Stability' or 'Dalle-2'"""

    remove_temp_files('/tmp')
    
    images_dir = tempfile.mkdtemp()
    dict = process_script(script)

    if model == 'Stability':
        
        for i, text in enumerate(dict['text_for_image_generation']):
            try:
                response = requests.post(
                f"https://api.stability.ai/v2beta/stable-image/generate/core",
                headers={
                    "authorization": os.environ.get('STABILITY_AI_API_KEY'),
                    "accept": "image/*"
                },
                files={"none": ''},
                data={
                    "prompt": text,
                    "output_format": "png",
                    'aspect_ratio': "9:16",
                },
                )
                print(f'image {i} generated')
                if response.status_code == 200:
                    with open(os.path.join(images_dir, f'image_{i}.png'), 'wb') as file:
                        file.write(response.content)
                else:
                    raise Exception(str(response.json()))
            except Exception as e:
                raise Exception(f"Image generation failed: {e}")
            
    elif model == 'Dalle-2':
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        for i, text in enumerate(dict['text_for_image_generation']):
            try:
                response = client.images.generate(
                    model="dall-e-2",
                    prompt=text,
                    size="1024x1024",
                    quality="standard",
                    n=1
                )
                image_url = response.data[0].url

                print(f'image {i} generated')
                # Download the image
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    with open(os.path.join(images_dir, f'image_{i}.png'), 'wb') as file:
                        file.write(image_response.content)
                else:
                    raise Exception(f"Failed to download image with status code {image_response.status_code} and message: {image_response.text}")

            except Exception as e:
                raise Exception(f"Image generation failed: {e}")

    return f'images are stored in {images_dir} directory'
    
def speech_generator(script):
    """
    Generates speech files for the given script using gTTS.
    Saves them to a temporary directory and returns the path.
    Args:
    script: a complete script containing narrations and image descriptions.
    """
    speeches_dir = tempfile.mkdtemp()

    dict = process_script(script)
    for i, text in enumerate(dict['text_for_speech_generation']):
        speech_path = generate_speech(text, num=i)
        print(f'speech {i} generated')
        os.rename(speech_path, os.path.join(speeches_dir, f'speech_{i}.mp3'))

    return f'speeches are stored in {speeches_dir} directory'

def find_temp_files(directory):
    temp_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path) and file_path.startswith('/tmp'):
                temp_files.append(file_path)

    return temp_files

def remove_temp_files(directory):
    temp_files = find_temp_files(directory)
    
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            print(f"Removed temp file: {temp_file}")
        except Exception as e:
            print(f"Error removing temp file {temp_file}: {e}")

def send_mail(user_mail, video_path):
    # Email configuration
    sender_email = 'prudhvisneha2003@gmail.com'
    receiver_email = user_mail
    password = 'pzxb drfj aebj ypuv'  # Normally, you should store sensitive information like passwords securely.

    # Create message container - the correct MIME type is multipart/alternative.
    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'From ShortsIn'
    msg['From'] = sender_email
    msg['To'] = receiver_email

    # Create the plain-text and HTML version of your message
    text = "Hello,"
    html = """\
    <html>
    <body>
        <p>Hello,

        Thank you for using ShortsIn.
        </p>
    </body>
    </html>
    """

    # Attach parts into message container
    part1 = MIMEText(text, 'plain')
    part2 = MIMEText(html, 'html')

    msg.attach(part1)
    msg.attach(part2)

    if os.path.isfile(video_path):
        with open(video_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename= {os.path.basename(video_path)}',
        )
        msg.attach(part)
    else:
        print(f"Error: The file {video_path} does not exist.")
        return

    # Initialize server variable
    server = None
    
    # Connect to the SMTP server
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)  # Example SMTP server and port
        server.starttls()  # Secure the connection
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        print('Email sent successfully!')
    except Exception as e:
        print(f'Error sending email: {str(e)}')
    finally:
        if server:
            server.quit()

def get_agents_and_tasks(groq_api_key, llm_choice):
    os.environ['GROQ_API_KEY'] = groq_api_key

    if llm_choice == "OpenAI":
        llm = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.6,
            max_tokens=1024,
            api_key=os.getenv('OPENAI_API_KEY')
        )
    elif llm_choice == "Gemini AI":
        genai.configure(api_key=os.getenv('OPENAI_API_KEY'))
        model = genai.GenerativeModel('gemini-pro')
        
        def gemini_generate(prompt):
            response = model.generate_content(prompt)
            return response.text

        llm = gemini_generate
    elif llm_choice == "g4f":
        llm = g4f.ChatCompletion.create(
            model="gpt-4",
            temperature=0.6,
            max_tokens=1024
        )

    script_agent = Agent(
        role='Senior Content Writer',
        goal='Craft engaging, concise, and informative narrations for YouTube short videos',
        backstory="""As a seasoned content writer, you excel at breaking down complex topics into captivating narratives that educate and entertain audiences. Your expertise lies in writing concise, attention-grabbing scripts for YouTube short videos.""",
        verbose=True,
        llm=llm,
        allow_delegation=False
    )

    image_descriptive_agent = Agent(
        role='Visual Storyteller',
        goal='Design stunning, contextually relevant visual descriptions for YouTube short videos. The number of descriptions should not be greater than three',
        backstory='With a keen eye for visual storytelling, you create compelling imagery that elevates the narrative and captivates the audience.',
        verbose=True,
        llm=llm,
        allow_delegation=False
    )

    img_speech_generating_agent = Agent(
        role='Multimedia Content Creator',
        goal='Generate high-quality images and speeches for YouTube short videos based on provided script',
        backstory='As a multimedia expert, you excel at creating engaging multimedia content that brings stories to life.',
        verbose=True,
        llm=llm,
        allow_delegation=False
    )

    editor = Agent(
        role = 'Video editor',
        goal = 'To make a video for YouTube shorts.',
        backstory = "You are a video editor working for a YouTube creator",
        verbose=True,
        llm=llm,
        allow_delegation = False,
        tools = [create_video_from_images_and_audio]
    )

    content_generation_task = Task(
        description='Generate engaging and informative content on the topic: {topic}. Use the
            description='Generate engaging and informative content on the topic: {topic}. Use the
                provided tool, only if you have no idea about the given topic.',
    expected_output="""Good corpus of text about: {topic}""",
    agent=script_agent,
    tools=[wiki_tool]
)

story_writing_task = Task(
    description='Write an engaging narration for a YouTube short video on the topic: {topic}',
    expected_output="""A short paragraph suitable for narrating in five seconds that provides an immersive experience to the audience. Follow the below example for output length and format.

    **Example input:**
    Ancient Wonders of the World

    **Output format:**
    Embark on a journey through time and marvel at the ancient wonders of the world! 
    From the majestic Great Pyramid of Giza, symbolizing the ingenuity of ancient Egypt, to the Hanging Gardens of Babylon, 
    These remarkable structures continue to intrigue and inspire awe, reminding us of humanity's enduring quest for greatness.
    """,
    agent=script_agent
)

img_text_task = Task(
    description='Given the narration, visually describe each sentence in the narration which will be used as a prompt for image generation.',
    expected_output="""Sentences encoded in <narration> and <image> tags. Follow the example below for the output format.

    **Example input:**
    Embark on a journey through time and marvel at the ancient wonders of the world! From the majestic Great Pyramid of Giza, symbolizing the ingenuity of ancient Egypt, to the Hanging Gardens of Babylon, an oasis of lush beauty amidst ancient Mesopotamia's arid landscape. These remarkable structures continue to intrigue and inspire awe, reminding us of humanity's enduring quest for greatness.

    **Output format:**

    <narration>Embark on a journey through time and marvel at the ancient wonders of the world!</narration>
    <image>A breathtaking view of various ancient wonders, showcasing their grandeur and mystery.</image>
    
    <narration>From the majestic Great Pyramid of Giza, symbolizing the ingenuity of ancient Egypt, to the Hanging Gardens of Babylon,</narration>
    <image>The majestic Great Pyramid of Giza standing tall, and the lush Hanging Gardens of Babylon amidst the arid landscape.</image>
    
    <narration>These remarkable structures continue to intrigue and inspire awe, reminding us of humanity's enduring quest for greatness.</narration>
    <image>Visitors captivated by the beauty and historical significance of these ancient marvels, exploring and marveling.</image>
    """,
    agent=image_descriptive_agent,
    context=[story_writing_task]
)

img_generation_task = Task(
    description='Given the script, use the given tool to generate images using {model}',
    expected_output="""Acknowledgement of images generation and path of images folder""",
    tools=[image_generator],
    context=[img_text_task],
    agent=img_speech_generating_agent
)

speech_generation_task = Task(
    description='Given the script, use the given tool to generate speech',
    expected_output="""Acknowledgement of speeches generation and path of speeches folder""",
    tools=[speech_generator],
    context=[img_text_task],
    agent=img_speech_generating_agent
)

make_video_task = Task(
    description='Create video using images and speeches from the folders received from previous task.',
    expected_output="output video path",
    agent=editor,
    context=[img_generation_task, speech_generation_task]
)

agents = [
    script_agent,
    image_descriptive_agent,
    img_speech_generating_agent,
    editor
]

tasks = [
    content_generation_task,
    story_writing_task,
    img_text_task,
    img_generation_task,
    speech_generation_task,
    make_video_task
]

return agents, tasks

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
    result = crew.kickoff(inputs={'topic': topic,'model': model})
    if 'tmp' in result:
        result = result.split('/')[-1]
    send_mail(user_mail, result)
    return result

intro = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8
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
            api = gr.Textbox(label='Enter your OpenAI API key here')
            stability_api = gr.Textbox(label='Enter your Stability AI API key here (optional)')
            email = gr.Textbox(label='Enter your email address here')
            btn = gr.Button('Generate', size='lg')

        with gr.Column():
            out = gr.Video(label='')

    btn.click(fn=generate_video, inputs=[inp, api, stability_api, email], outputs=out)

app.launch()
