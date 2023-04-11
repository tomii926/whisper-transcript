import whisper

model = whisper.load_model("large-v2")
result = model.transcribe("sample.wav")
print(result["text"])

with open('transcript-v2.txt', 'w') as f:
    print(result["text"], file=f)