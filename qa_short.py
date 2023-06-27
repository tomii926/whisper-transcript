import os

import openai
from dotenv import load_dotenv
from api import ask_question

from transcript import transcript

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]


text = transcript("data/struggle9.mp3")


def gen_prompt(q):
  prompt = f"""以下の情報は、NTTコミュニケーションズによる営業の商談に関するまとめである。
```
{text}
```

商談の内容に対して、以下の質問がある。
質問: {q}

先ほどの文章を手がかりにして、質問に一言で回答してください。わからない場合は、「不明」と答えてください。
回答: """
  return prompt


with open('questions.txt', 'r') as f:
    queries = f.readlines()
    for query in queries:
        query = query.strip()
        print(query)
        print(ask_question(gen_prompt(query)))
        print('')
