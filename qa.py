from dotenv import load_dotenv

from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator


load_dotenv()

loader = TextLoader('transcript-v2.txt')

index = VectorstoreIndexCreator().from_loaders([loader])

query = """
話者の名前は？
"""
print(index.query(query, chain_type="map_reduce"))

