from pyannote.audio import Pipeline
import os

token = os.environ["HUGGINGFACE_API_KEY"]
print(token)

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="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")
diarization = pipeline("sample.wav")

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")


import whisper
model = whisper.load_model("large")

from pyannote.audio import Audio

audio_file = "audio.wav"
diarization = pipeline(audio_file)

audio = Audio(sample_rate=16000, mono=True)

for segment, _, speaker in diarization.itertracks(yield_label=True):
    waveform, sample_rate = audio.crop(audio_file, segment)
    text = model.transcribe(waveform.squeeze().numpy())["text"]
    print(f"[{segment.start:03.1f}s - {segment.end:03.1f}s] {speaker}: {text}")