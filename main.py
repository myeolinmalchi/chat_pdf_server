from chromadb.config import Settings

import os
from dotenv import load_dotenv

from fastapi import Depends, FastAPI
from typing import List, Optional

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma

from chat import make_chain, query_papers
from pydantic import BaseModel
from db import auth

load_dotenv()

############## Chroma DB ##############
CHROMA_DB_HOST = os.environ['CHROMA_DB_HOST']
CHROMA_DB_PORT = os.environ['CHROMA_DB_PORT']
COLLECTION_NAME = os.environ['CHROMA_DB_COLLECTION']

client_settings = Settings(
    chroma_api_impl="rest", 
    chroma_server_host=CHROMA_DB_HOST, 
    chroma_server_http_port=CHROMA_DB_PORT
)

vectorstore = Chroma(client_settings=client_settings, collection_name=COLLECTION_NAME)
############## Chroma DB ##############

############## Embedding ##############
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
############## Embedding ##############

############## OpenAI ##############
openai = ChatOpenAI(
    model='gpt-4', 
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

@app.get("/api/v1/docs/query/{query}", dependencies=[Depends(auth)])
async def query_docs(query: str, topk: Optional[int] = None):
    embedded_query  = embeddings.embed_query(query)
    papers = query_papers(vectorstore, embedded_query=embedded_query, top_k=topk or 5)

    return QueryDocResponse(docs=[DocInfo(doc_idx=int(doc_idx), doc_title=doc_title) for (doc_idx, doc_title) in papers])

class Chat(BaseModel):
    question: str
    answer: str

class CompletionRequest(BaseModel):
    chat_history: List[Chat]
    new_question: str

class CompletionResponse(BaseModel):
    answer: str

@app.post("/api/v1/docs/{doc_idx}/completion", dependencies=[Depends(auth)])
async def qa(doc_idx: int, body: CompletionRequest) -> CompletionResponse:
    chain = make_chain(vectorstore, openai, doc_idx)
    question = body.new_question
    chat_history = [(history.question, history.answer) for history in body.chat_history]
    result = chain({"question": question, "chat_history": chat_history})

    return CompletionResponse(answer=result['answer']) 
