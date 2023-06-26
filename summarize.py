from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys

load_dotenv()

loader = TextLoader('transcript.txt')

documents = loader.load()

# documentsを分割する。1000文字ずつ。ただし50文字ずつoverlapさせる。
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50, separators=["\n", "。", "、", " ", ])
texts = text_splitter.split_documents(documents)

prompt_template = """以下の文章は、NTTコミュニケーションズによる営業の商談の音声を文字起こししたデータの一部です。このデータを見て、商談の内容をまとめてください。まとめた結果は長くなってもいいので、会話の情報をできるだけ失わないように注意してください。:

「{text}」

まとめた結果: """

PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

llm = OpenAI(max_tokens=1300) # プロンプトと合わせて4097tokenを超えないようにいするとこれくらいが限界(最後のcombineがたぶんボトルネックになる)

chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=PROMPT, combine_prompt=PROMPT)

summarized_data = chain.run(texts)
print(summarized_data)
