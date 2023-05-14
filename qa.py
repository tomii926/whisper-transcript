from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

load_dotenv()

loader = TextLoader('transcript.txt')
index = VectorstoreIndexCreator().from_loaders([loader])

with open('questions.txt', 'r') as f:
    queries = f.readlines()
    for query in queries:
        query = query.strip()
        print(query)
        query += '日本語で答えてください。'
        print(index.query(query, chain_type="map_reduce"))
        print('')

