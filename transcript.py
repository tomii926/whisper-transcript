import os

import openai
from dotenv import load_dotenv
from pydub import AudioSegment


def transcript(file_path):
    audio = AudioSegment.from_file(file_path, "m4a")

    chunk = 20 * 60 * 1000

    final = ""
    
    for i in range(0, len(audio), chunk):
        # 10分ごとにファイルを保存する。

        tmp_file_path = f"tmp/{i}.mp3"
        audio[i:i+chunk].export(tmp_file_path, format="mp3")

        # 分割したファイルをOpenAI APIに投げる。
        with open(tmp_file_path, "rb") as f:    
            transcript = openai.Audio.transcribe("whisper-1", f)

        text = transcript['text']
        final += text

    # ./tmp以下のファイルを消去する
    for file in os.listdir("./tmp"):
        os.remove(f"./tmp/{file}")

    return final


if __name__ == "__main__":
    load_dotenv()

    # APIキーの設定
    openai.api_key = os.environ["OPENAI_API_KEY"]

    print(transcript("sample.m4a"))
