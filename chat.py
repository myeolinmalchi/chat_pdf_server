from typing import List, Tuple
from chromadb.api import Collection
from langchain import LLMChain, PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
from utils import CustomVectorStoreRetriever
from langchain.chains import ConversationalRetrievalChain


############## 유사도 순으로 문서 검색 ##############

def query_papers(collection: Collection,
                 embedded_query: List[float],
                 top_k: int) -> List[Tuple[str, str]]:
    top_results = collection.query(
        query_embeddings=embedded_query, 
        n_results=top_k*3, 
        include=["metadatas"], 
    )

    if top_results["metadatas"] is not None:
        doc_idxs = [result["doc_idx"] for result in top_results["metadatas"][0]]
        doc_titles = [result["doc_title"] for result in top_results["metadatas"][0]]

        recommended_papers = []
        seen_paper_idxs = set()
        for i in range(len(doc_idxs)):
            doc_idx = doc_idxs[i]
            file_name = doc_titles[i]

            if doc_idx not in seen_paper_idxs:
                seen_paper_idxs.add(doc_idx)
                recommended_papers.append((doc_idx, file_name))

            if(len(recommended_papers)) == top_k:
                break

        return recommended_papers

    return []

############## 유사도 순으로 문서 검색 ##############


############## Setup QA Chain ##############

CONDENSE_PROMPT = PromptTemplate(
template='''
You must answer in Korean.
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:''', input_variables=["chat_history", "question"])

QA_PROMPT = PromptTemplate(
template='''
You must answer in Korean.
You are an AI assistant providing helpful advice. You are given the following extracted parts of a long document and a question. Provide a conversational answer based on the context provided.
You should only provide hyperlinks that reference the context below. Do NOT make up hyperlinks.
If you can't find the answer in the context below, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

Question: {question}
=========
{context}
=========
Answer in Markdown:''', input_variables=["question", "context"])

def make_chain(vectorstore: Chroma, llm, doc_idx): 
    question_generator = LLMChain(
        llm=llm, 
        prompt=CONDENSE_PROMPT, 
        verbose=True
    )

    doc_chain = load_qa_chain(
        llm=llm, 
        prompt=QA_PROMPT, 
        verbose=True
    )

    return ConversationalRetrievalChain(
        retriever=CustomVectorStoreRetriever(
            vectorstore=vectorstore, 
            search_kwargs={
                "filter": {
                    "doc_idx": doc_idx
                }, 
                "k": 3
            }, 
            search_type="similarity"
            ), 
        combine_docs_chain=doc_chain, 
        question_generator=question_generator, 
        return_source_documents=True, 
        verbose=True, 
    )

############## Setup QA Chain ##############
