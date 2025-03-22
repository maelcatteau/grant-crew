import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv


## Load the API keys from the .env file en vérifiant leur présence
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")
if not llama_cloud_api_key:
    raise ValueError("LLAMA_CLOUD_API_KEY is not set in the .env file")

# Exemple simple avec un document en mémoire
text = "L'auteur a grandi en travaillant sur l'écriture et la programmation. Il a écrit des nouvelles et a également essayé de programmer sur un ordinateur IBM 1401 en utilisant une ancienne version de Fortran."
documents = [Document(text=text)]

llm = OpenAI(api_key=openai_api_key)
embed_model = OpenAIEmbedding(api_key=openai_api_key)

index = VectorStoreIndex.from_documents(documents, llm=llm, embed_model=embed_model)
query_engine = index.as_query_engine()
query = "Qu'est-ce que l'auteur faisait en grandissant ?"
response = query_engine.query(query)
print(f"Question : {query}")
print(f"Réponse : {response}")