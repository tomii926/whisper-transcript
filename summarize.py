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

# summarized_data = chain.run(texts)
summarized_data = """NTTコミュニケーションズと顧客の間で営業の商談が行われ、顧客はMacを使用し、NumbersというExcelのようなソフトを使って請求書の作成を行っていることが分かりました。NTTコミュニケーションズは、Link Print Cloudを使って請求書を発行する企業に対し、請求書を貼り付けたメールを顧客に送信するサービスを提供しています。その他のサービス内容として、郵送の代行、FAXの代行、100種類以上のフォーマットを本社で用意していること、CSVデータを読み込みマイメージを作成し、サーバー代を記載する必要があること、インブエス制度や電子帳簿保存法への対応、権限設定により自分が作った資料しか見れないような設定ができることなどがあります。サービスの年額は26万4千円で初期費用は11万円であり、オプション料金も発生します。クライアントは10月のインボイス制度開始や来年1月の電子情報保存法開始を考慮して早めの導入を検討していることが分かりました。商談の後、NTTコミュニケーションズは見積書を作成して顧客にお送りする予定であり、検討状況をメールまたは電話でお伺いすることを約束しています。"""

print(summarized_data)

from api import ask_question

with open('questions.txt') as f:
    questions = f.readlines()
    for q in questions:
        print(q)
        prompt = f"""以下の文章は、NTTコミュニケーションズによる営業の商談の内容の要約です。

```{summarized_data}```

この文章に対して、次の質問に一言で回答してください。わからない場合は、「不明」とだけ答えてください。

質問：{q}
"""
        print(ask_question(prompt))
        print("")

