from chromadb.config import Settings
import json

import os
from dotenv import load_dotenv

from fastapi import Depends, FastAPI, HTTPException, status
from typing import List, Optional
from googletrans import Translator
from googletrans.client import Translated
import openai

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma

from pydantic import BaseModel

from chat import make_chain, query_papers
from db import auth
from script import summarize_doc

load_dotenv()

translator = Translator()

############## Embedding ##############
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
############## Embedding ##############

############## Chroma DB ##############
CHROMA_DB_HOST = os.environ['CHROMA_DB_HOST']
CHROMA_DB_PORT = os.environ['CHROMA_DB_PORT']
COLLECTION_NAME = os.environ['CHROMA_DB_COLLECTION']

client_settings = Settings(
    chroma_api_impl="rest", 
    chroma_server_host=CHROMA_DB_HOST, 
    chroma_server_http_port=CHROMA_DB_PORT
)

vectorstore = Chroma(
    client_settings=client_settings,
    collection_name=COLLECTION_NAME, 
    embedding_function=embeddings
)
############## Chroma DB ##############


############## OpenAI ##############
openai.api_key = os.environ["OPENAI_API_KEY"]
_openai = ChatOpenAI(
    model='gpt-3.5-turbo-16k', 
    max_tokens=2048, 
    client=None, 
    temperature=0.73, 
)
############## OpenAI ##############

############## Models ##############
class DocInfo(BaseModel):
    doc_id: int
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

class SummarizeResponse(BaseModel):
    script: str

class ClassificationRequest(BaseModel):
    messages: List[Message]

class ClassificationResponse(BaseModel):
    type: str
    answer: Optional[str] = None
    docs: Optional[List[DocInfo]] = None

############## Models ##############

app = FastAPI()

@app.get('/', status_code=status.HTTP_200_OK)
def healthcheck():
    return { 'message': 'Everything OK!' }

@app.get("/api/v1/docs/search", dependencies=[Depends(auth)])
async def query_docs(query: str, topk: Optional[int] = None):
    papers = query_papers(query=query, vectorstore=vectorstore, top_k=topk or 5, translator=translator)
    return QueryDocResponse(docs=[DocInfo(doc_id=int(doc_id), doc_title=doc_title) for (doc_id, doc_title) in papers])

@app.get("/api/v1/docs/{doc_id}/summarize", dependencies=[Depends(auth)])
async def summarize(doc_id: int):
    script = summarize_doc(vectorstore=vectorstore, translator=translator, doc_id=doc_id)
    return SummarizeResponse(script=script)

@app.post("/api/v1/docs/{doc_id}/completion", dependencies=[Depends(auth)])
async def qa(doc_id: int, body: CompletionRequest) -> CompletionResponse:
    chain = make_chain(vectorstore, _openai, doc_id)
    question = body.new_question
    translated_question = translator.translate(text=question, src="ko", dest="en")
    if isinstance(translated_question, Translated):
        question = translated_question.text
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
                "name": "search_papers", 
                "description": "Search for documentations related to the user's question.", 
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "type": {
                            "type": "string", 
                            "enum": ["QA", "TD"]
                        }, 
                        "query": {
                            "type": "string", 
                            "description": "Answer the user's question about the most recent state of the union address"
                        }
                    }, 
                    "required": ["query", "type"]
                }, 
            }, 
        ], 
        function_call="auto"
    )

    if "choices" in completion:
        message = completion["choices"][0]["message"]

        # 문서 검색인 경우
        if "function_call" in message and message["function_call"]["name"] == "search_papers":
            jsonstring = message["function_call"]["arguments"]
            args = json.loads(jsonstring)
            query = args["query"]
            papers = query_papers(query=query,vectorstore=vectorstore, top_k=5, translator=translator)
            docs = [DocInfo(doc_id=int(doc_id), doc_title=doc_title) 
                    for (doc_id, doc_title) in papers]

            return ClassificationResponse(type="TD", docs=docs)

        # 단순 응답인 경우
        else:
            answer = message["content"]
            return ClassificationResponse(type="QA", answer=answer)

    else:
        raise HTTPException(400)




    
