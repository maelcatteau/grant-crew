import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.ingestion import IngestionPipeline
import pinecone
from pinecone import Pinecone


## Load the variables from the .env file en vérifiant leur présence
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")
if not llama_cloud_api_key:
    raise ValueError("LLAMA_CLOUD_API_KEY is not set in the .env file")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is not set in the .env file")
if not pinecone_environment:
    raise ValueError("PINECONE_ENVIRONMENT is not set in the .env file")
if not pinecone_index_name:
    raise ValueError("PINECONE_INDEX_NAME is not set in the .env file")

## Initialisation de la connexion à Pinecone
pc = Pinecone(api_key=pinecone_api_key)
embedding_dimension = 1536

## Initialisation du modèle d'embedding avec le choix du modèle
embed_model = OpenAIEmbedding(api_key=openai_api_key, model="text-embedding-3-small")


# Initialisation du vector store pinecone
vector_store = PineconeVectorStore(
    pinecone_index=pc.Index(pinecone_index_name, environment=pinecone_environment),
    embedding_dim=embedding_dimension,
)

# Initialisation du text splitter
splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model, include_metadata=False
)
# Création du pipeline d'ingestion
pipeline = IngestionPipeline(
    transformations=[splitter, embed_model],
    vector_store=vector_store
)

##Chargement des données à partir du répertoire data
reader = SimpleDirectoryReader(
    input_dir="./data",
    file_extractor={
        ".pdf": LlamaParse(
            api_key=llama_cloud_api_key,
            result_type="markdown",
            num_workers=4,
            verbose=True,
            language="fr",
        )
    }
)
documents = reader.load_data()

#Execution du pipeline d'ingestion 
nodes = pipeline.run(documents=documents)
print(f"Nombre de noeuds ingérés : {len(nodes)}")
print(nodes[1].get_content())

# Création de l'index
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)