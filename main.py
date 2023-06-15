from chromadb.config import Settings
import json

import os
from dotenv import load_dotenv

from fastapi import Depends, FastAPI, status
from typing import List, Optional
import openai

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma

from pydantic import BaseModel

from chat import make_chain, query_papers
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
openai.api_key = os.environ["OPENAI_API_KEY"]
_openai = ChatOpenAI(
    model='gpt-3.5-turbo-0613', 
    max_tokens=512, 
    client=None
)
############## OpenAI ##############

############## Models ##############
class DocInfo(BaseModel):
    doc_idx: int
    doc_title: str

class QueryDocResponse(BaseModel):
    docs: List[DocInfo]

class Chat(BaseModel):
    question: str
    answer: str

class CompletionRequest(BaseModel):
    chat_history: List[Chat]
    new_question: str

class CompletionResponse(BaseModel):
    answer: str

class Message(BaseModel):
    role: str
    content: str

class ClassificationRequest(BaseModel):
    messages: List[Message]

class ClassificationResponse(BaseModel):
    type: str
    answer: Optional[str]
    docs: List[DocInfo]

############## Models ##############

app = FastAPI()

@app.get('/', status_code=status.HTTP_200_OK)
def healthcheck():
    return { 'message': 'Everything OK!' }


@app.get("/api/v1/docs/search", dependencies=[Depends(auth)])
async def query_docs(query: str, topk: Optional[int] = None):
    embedded_query  = embeddings.embed_query(query)
    papers = query_papers(vectorstore, embedded_query=embedded_query, top_k=topk or 5)

    return QueryDocResponse(docs=[DocInfo(doc_idx=int(doc_idx), doc_title=doc_title) for (doc_idx, doc_title) in papers])


@app.post("/api/v1/docs/{doc_idx}/completion", dependencies=[Depends(auth)])
async def qa(doc_idx: int, body: CompletionRequest) -> CompletionResponse:
    chain = make_chain(vectorstore, _openai, doc_idx)
    question = body.new_question
    chat_history = [(history.question, history.answer) for history in body.chat_history]
    result = chain({"question": question, "chat_history": chat_history})

    return CompletionResponse(answer=result['answer']) 


@app.post("/api/v1/classification", dependencies=[Depends(auth)])
async def classification(body: ClassificationRequest):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613", 
        messages=[{
            "role": "system", 
            "content": "Answer in Korean",
        }]+[message.dict() for message in body.messages], 
        functions=[
            {
                "name": "qa_or_td", 
                "description": "Provide keywords or search terms to answer the user's question, or to find the documentation needed to answer the question.", 
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "type": {
                            "type": "string", 
                            "description": "Answer 'TD' when you need access to external documentation to answer a user's question, or 'QA' for simple questions. Answer in Korean."
                        }, 
                        "answer": {
                            "type": "string", 
                            "answer": "If the type is TD, make a list of keywords related to the user's question. If the type is 'QA', write a response to the user's last question."
                        }
                    }, 
                    "required": ["type", "answer"]
                }, 
            }
        ], 
        function_call={"name": "qa_or_td"}
    )

    jsonstring = completion["choices"][0]["message"]["function_call"]["arguments"]
    response = json.loads(jsonstring)

    if response["type"] == "TD":
        query = response["answer"]
        embedded_query = embeddings.embed_query(query)
        papers = query_papers(vectorstore, embedded_query=embedded_query, top_k=5)
        docs = [DocInfo(doc_idx=int(doc_idx), doc_title=doc_title) 
                for (doc_idx, doc_title) in papers]

        return ClassificationResponse(type="TD", answer=response["answer"], docs=docs)

    return ClassificationResponse(type=response["type"], answer=response["answer"], docs=[])

    
