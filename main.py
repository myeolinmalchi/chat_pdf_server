import chromadb
from chromadb.config import Settings

import os
from dotenv import load_dotenv

from fastapi import FastAPI
from typing import List, Optional

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma

from chat import make_chain, query_papers

from pydantic import BaseModel

############## Chroma DB ##############
load_dotenv()

CHROMA_DB_HOST = os.environ['CHROMA_DB_HOST']
CHROMA_DB_PORT = os.environ['CHROMA_DB_PORT']

client_settings = Settings(
    chroma_api_impl="rest", 
    chroma_server_host=CHROMA_DB_HOST, 
    chroma_server_http_port=CHROMA_DB_PORT
)
chroma_client = chromadb.Client(client_settings)
collection = chroma_client.get_collection(name="test_collection")

vectorstore = Chroma(client_settings=client_settings, collection_name="test_collection")
############## Chroma DB ##############

############## Embedding ##############
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
############## Embedding ##############

############## OpenAI ##############
openai = ChatOpenAI(
    model='gpt-3.5-turbo', 
    max_tokens=512, 
    client=None
)
############## OpenAI ##############

app = FastAPI()

class DocInfo(BaseModel):
    doc_idx: int
    doc_title: str

class QueryDocResponse(BaseModel):
    docs: list[DocInfo]

@app.get("/api/v1/docs/query/{query}")
async def _query_docs(query: str, topk: Optional[int] = None):
    embedded_query  = embeddings.embed_query(query)
    papers = query_papers(collection, embedded_query=embedded_query, top_k=topk or 5)

    return QueryDocResponse(docs=[DocInfo(doc_idx=int(doc_idx), doc_title=doc_title) for (doc_idx, doc_title) in papers])

class Chat(BaseModel):
    question: str
    answer: str

class CompletionRequest(BaseModel):
    chat_history: List[Chat]
    new_question: str

class CompletionResponse(BaseModel):
    answer: str

@app.post("/api/v1/docs/{doc_idx}/completion")
async def qa(doc_idx: int, body: CompletionRequest) -> CompletionResponse:
    chain = make_chain(vectorstore, openai, doc_idx)
    question = body.new_question
    print(question)
    chat_history = [(history.question, history.answer) for history in body.chat_history]
    result = chain({"question": question, "chat_history": chat_history})

    return CompletionResponse(answer=result['answer']) 
