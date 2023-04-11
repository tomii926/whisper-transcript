import os

import whisper
from pyannote.audio import Audio, Pipeline


token = os.environ["HUGGINGFACE_API_KEY"]
audio_file = 'sample.wav'

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=token)

print("How many speakers?")
num_speakers = int(input())
diarization = pipeline(audio_file, num_speakers=num_speakers)

model = whisper.load_model("large")

audio = Audio(sample_rate=16000, mono=True)

with open('result.txt', 'w') as f:
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        waveform, sample_rate = audio.crop(audio_file, segment)
        text = model.transcribe(waveform.squeeze().numpy())["text"]
        log = f"[{segment.start:03.1f}s - {segment.end:03.1f}s] {speaker}: {text}"
        log = f"{speaker}: {text}"
        print(log, file=f)
        print(log)
