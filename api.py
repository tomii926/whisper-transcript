import os

import openai
from dotenv import load_dotenv
load_dotenv()

# APIキーの設定
openai.api_key = os.environ["OPENAI_API_KEY"]

def ask_question(question):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0]["message"]["content"].strip()


if __name__ == '__main__':
    print(ask_question("こんにちは"))