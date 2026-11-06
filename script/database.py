from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from qdrant_client.models import (
    Distance,
    VectorParams,
)
from qdrant_client import QdrantClient
import pandas as pd
from dotenv import load_dotenv
import os 
from uuid import uuid4
import logging
from langchain_qdrant import QdrantVectorStore

load_dotenv()
logging.basicConfig(level=logging.INFO)

logging.info("Initializing Qdrant client...")
qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=100,  # Increase timeout here
            prefer_grpc=False,  # Set to True if using gRPC port 6334
)
        
embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

def load_data(csv_path: str, limit=1) -> list[Document]:
    logging.info(f"Loading data from {csv_path} with limit {limit}...")
    df = pd.read_csv(csv_path)
    df = df.head(limit)

    documents = []

    for idx, row in df.iterrows():
        # 1. Combine selected columns for embedding
        content = "\n".join(
            f"{col}: {row.get(col, '')}"
            for col in [
                'Series_Title',
                'Released_Year',
                'Genre',
                'Runtime',
                'Overview',
                'Star1',
                'Star2',
                'Star3',
                'Star4',
                'Director'
            ]
            if pd.notna(row.get(col))
            and str(row.get(col)).strip()
        )

        if not content.strip():
            continue

        # 2. Combine actors into a list
        cast = [
            str(row.get(col)).strip()
            for col in ['Star1', 'Star2', 'Star3', 'Star4']
            if pd.notna(row.get(col))
            and str(row.get(col)).strip()
        ]

        # 3. Create document with metadata
        doc = Document(
            page_content=content,
            metadata={
                'title': str(row.get('Series_Title', 'Unknown')),
                'released_year': str(row.get('Released_Year', 'Unknown')),
                'runtime': str(row.get('Runtime', 'Unknown')),
                'genre': str(row.get('Genre', 'Unknown')),
                'imdb_rating': float(row.get('IMDB_Rating', 0) or 0),
                'cast': cast,
                'director': str(row.get('Director', 'Unknown')),
                'gross': str(row.get('Gross', 'Unknown')),
                'row_index': int(idx),
                'source': 'imdb_top_1000.csv'
                }
        )
        documents.append(doc)
    logging.info(f"Loaded {len(documents)} documents.")
    return documents

def main():
    documents = load_data('dataset/imdb_top_1000.csv', limit=1000)

    collections = qdrant_client.get_collections().collections
    collection_names = [col.name for col in collections]

    if not os.getenv("QDRANT_COLLECTION_NAME") in collection_names:
        logging.info(f"Creating collection {os.getenv('QDRANT_COLLECTION_NAME')}...")
        qdrant_client.create_collection(
            collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    
    uuids = [str(uuid4()) for _ in range(len(documents))]
    logging.info(f"Adding {len(documents)} documents to the vector store...")
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embedding=embeddings,
    )
    vector_store.add_documents(documents=documents, ids=uuids)
    logging.info("Documents added successfully.")

if __name__ == "__main__":
    main()