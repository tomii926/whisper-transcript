import whisper
# 普通のモデル
model = whisper.load_model("base")
result = model.transcribe("sample.m4a")
print(result["text"])

print('')
# 強いモデル
model_large = whisper.load_model("large")
result = model_large.transcribe("sample.m4a")
print(result['text'])
