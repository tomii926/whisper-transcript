from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys

load_dotenv()

loader = TextLoader('transcript.txt')

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

from langchain.vectorstores import Chroma

db = Chroma.from_documents(texts, embeddings)

retriever = db.as_retriever()


question_prompt_template = """以下の文章は、長い商談の音声データを書き起こしたものの一部である。 
```
{context}
```
この文章に対して、次のような質問がある。
質問: {question}

質問に答える際に参考になるであろう文章を、先ほどの文章からできるだけ多く取り出す。
取り出した文章は以下の通り。:"""


QUESTION_PROMPT = PromptTemplate(
    template=question_prompt_template, input_variables=["context", "question"]
)


combine_prompt_template = """以下の複数個の文章は、ある長い商談の音声データの書き起こしを、複数箇所抜粋したものである。
```
{summaries}
```
商談の内容に対して、以下のような質問がある。
質問: {question}

先ほどの文章の全てを手がかりにして、質問に一言で答える。
答え:"""

COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template, input_variables=["summaries", "question"]
)


chain = load_qa_chain(llm=OpenAI(), chain_type="map_reduce", return_intermediate_steps=True, question_prompt=QUESTION_PROMPT, combine_prompt=COMBINE_PROMPT)

if len(sys.argv) >= 2:
    query = sys.argv[1]
    print(query)
    docs = retriever.get_relevant_documents(query)
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
