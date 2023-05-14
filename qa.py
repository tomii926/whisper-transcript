from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

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

chain = load_qa_chain(llm=OpenAI(), chain_type="map_reduce", return_intermediate_steps=True)


with open('questions.txt', 'r') as f:
    queries = f.readlines()
    for query in queries:
        query = query.strip()
        print(query)
        docs = retriever.get_relevant_documents(query)
        res = chain({"question": query, "input_documents": docs})
        print(res)
        print('')

