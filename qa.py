import os
import sys

import openai
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from transcript import transcript

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n", "。", "、", " ", ])

text = transcript("data/struggle7.m4a")

with open("log/struggle7.txt", "w") as f:
    f.write(text)

texts = text_splitter.split_text(text)

embeddings = OpenAIEmbeddings()

db = Chroma.from_texts(texts, embeddings=embeddings)

retriever = db.as_retriever()


question_prompt_template = """以下の文章は、NTTコミュニケーションズによる営業の商談の音声の書き起こしデータの一部である。 
```
{context}
```
この文章に対して、次のような質問がある。
質問: {question}

先ほどの文章から、上記の質問に答える際に少しでも関係がありそうな部分を、できるだけ多く取り出す。
取り出した文章は以下の通り。

"""


QUESTION_PROMPT = PromptTemplate(
    template=question_prompt_template, input_variables=["context", "question"]
)


combine_prompt_template = """以下の複数個の文章は、NTTコミュニケーションズによる営業の商談の音声の書き起こしデータを、複数箇所抜粋したものである。
```
{summaries}
```
商談の内容に対して、以下の質問がある。
質問: {question}

先ほどの文章を手がかりにして、質問に一言で回答してください。
回答: """

COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template, input_variables=["summaries", "question"]
)

chain = load_qa_chain(llm=OpenAI(), chain_type="map_reduce", return_intermediate_steps=True, question_prompt=QUESTION_PROMPT, combine_prompt=COMBINE_PROMPT)

if len(sys.argv) >= 2:
    query = sys.argv[1]
    print(query)
    docs = retriever.get_relevant_documents(query)
    print(docs)
    print(len(docs))
    res = chain({"question": query, "input_documents": docs})
    print(res['output_text'])
    exit(0)

with open('questions.txt', 'r') as f:
    queries = f.readlines()
    for query in queries:
        query = query.strip()
        print(query)
        docs = retriever.get_relevant_documents(query)
        res = chain({"question": query, "input_documents": docs})
        # print(res['intermediate_steps'])
        print(res['output_text'])
        print('')
