import whisper

model = whisper.load_model("large-v2")
result = model.transcribe("sample.m4a")
print(result["text"])

with open('transcript-v2.txt', 'w') as f:
    print(result["text"], file=f)