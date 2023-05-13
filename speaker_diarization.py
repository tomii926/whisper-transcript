import os

import whisper
from dotenv import load_dotenv
from pyannote.audio import Audio, Pipeline
import json


load_dotenv()

token = os.environ["HUGGINGFACE_API_KEY"]
audio_file = 'struggle.wav'

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=token)

print("How many speakers?")
num_speakers = int(input())
diarization = pipeline(audio_file, num_speakers=num_speakers)

model = whisper.load_model("large")

audio = Audio(sample_rate=16000, mono=True)

with open('diarization.json', 'w') as f:
    json.dump(diarization.for_json(), f)
