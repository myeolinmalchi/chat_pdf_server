import os

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from kss import split_sentences
from utils import create_chunks
from langchain.embeddings import SentenceTransformerEmbeddings

class EmbeddingFailureException(Exception):
    def __init__(self):
        super().__init__('임베딩에 실패했습니다.')

def load_pdf_files(path):
    files = os.listdir(path)
    pdf_files = [file for file in files if file.endswith('.pdf')]
    return pdf_files

def embed_pdf(path, idx, file_name, embeddings, collection_name, client_settings):
    file_path = os.path.join(path, file_name)
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    temp = split_sentences(
        text=[page.page_content for page in pages], 
        strip=False, 
    )

    sentences = sum(temp, [])
    chuncks = create_chunks(sentences)

    Chroma.from_texts(
        chuncks, 
        embedding=embeddings, 
        collection_name=collection_name, 
        metadatas=[{"doc_title": file_name, "doc_idx": idx} for _ in range(len(chuncks))], 
        documents=[file_name for _ in range(len(chuncks))], 
        client_settings=client_settings
    )

def embed_dataset(path, chroma_client, collection_name, embeddings, client_settings):
    try:
        pdf_files = load_pdf_files(path=path)

        if collection_name in [collection.name for collection in chroma_client.list_collections()]:
            print(f"Collection '{collection_name}'이 이미 존재하여 삭제합니다.")
            chroma_client.delete_collection(name=collection_name)

        print(f"Collection '{collection_name}'을 생성합니다.")
        chroma_client.create_collection(name=collection_name)

        for idx, file_name in enumerate(pdf_files):
            print(f"Embedding [{idx+1} / {len(pdf_files)}] ... ", end="")
            embed_pdf(
                idx=idx, 
                file_name=file_name, 
                embeddings=embeddings, 
                collection_name=collection_name, 
                path=path, 
                client_settings=client_settings
            )
            print("Done.")

        print("Complete.")

    except Exception as e:
        print(f"Exception: {e}")
        print(f"오류가 발생하여 Collection을 초기화합니다.")
        try:
            chroma_client.delete_collection(name=collection_name)
        except Exception as e:
            print(f"Exception: {e}")
            print("Collection 초기화에 실패하였습니다.")
            raise EmbeddingFailureException

load_dotenv()

CHROMA_DB_HOST = os.environ['CHROMA_DB_HOST']
CHROMA_DB_PORT = os.environ['CHROMA_DB_PORT']

client_settings = Settings(
    chroma_api_impl="rest", 
    chroma_server_host=CHROMA_DB_HOST, 
    chroma_server_http_port=CHROMA_DB_PORT
)

print(f"Chroma DB에 연결합니다. [{CHROMA_DB_HOST}:{CHROMA_DB_PORT}]\n")

chroma_client = chromadb.Client(client_settings)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

embed_dataset(
    path='./datasets', 
    chroma_client=chroma_client, 
    collection_name='test_collection', 
    embeddings=embeddings, 
    client_settings=client_settings
)
