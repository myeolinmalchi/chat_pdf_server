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
    model='gpt-4', 
    max_tokens=512, 
    client=None
)
############## OpenAI ##############

############## Models ##############
class DocInfo(BaseModel):
    doc_idx: int
    doc_title: str

class QueryDocResponse(BaseModel):
    docs: list[DocInfo]

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
    answer: str
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
    print(body.messages[0].dict())
    completion = openai.ChatCompletion.create(
        model="gpt-4", 
        messages=[{"role": "system", "content": """As an AI Assistant, you must answer different types of questions depending on the type of question the user has. There are two types of questions: QA (Question Answering) and TD (QA About Techincal Documentation). QA is for general queries, while TD means that the user's question requires specific information about a particular technology, so you need to search for detailed documentation and include additional context. 

You must answer in the following JSON string format, and do not include any additional comments or information outside of the JSON string:

{
    "type": "QA or TD",
    "answer": "If type is TD, write a sentence or keyword to search for the appropriate article in the context of the conversation so far. If type is QA, answer the user's last question."
}
"""}] + [message.dict() for message in body.messages]
    )


    jsonstring = completion["choices"][0]["message"]["content"]
    print(jsonstring)
    _json = json.loads(jsonstring)
    return ClassificationResponse(type=_json["type"], answer=_json["answer"])

    
