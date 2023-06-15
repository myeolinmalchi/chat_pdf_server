from typing import List, Tuple
from googletrans import Translator
from googletrans.client import Translated
from langchain import LLMChain, PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
from utils import CustomVectorStoreRetriever
from langchain.chains import ConversationalRetrievalChain


############## 문서 검색 ##############

def query_papers(
        query: str,
        vectorstore: Chroma,
        top_k: int, 
        translator: Translator, 
    ) -> List[Tuple[str, str]]:
    # 질문(query)을 영어로 번역합니다.
    translated_query = translator.translate(text=query, src="ko", dest="en")
    if isinstance(translated_query, Translated):
        query = translated_query.text

    # top_k의 3배만큼 chunk를 추출합니다.
    results = vectorstore.similarity_search_with_score(query, k=top_k*3)
    docs = [(result.metadata["doc_id"], result.metadata["doc_title"]) for [result, _] in results]

    # docs에 대하여 루프를 돌면서 문서가 중복되지 않게끔 추출합니다.
    # 이 과정에서 총 문서의 수가 top_k보다 적을 수 있습니다.
    recommended_papers = []
    seen_paper_idxs = set()
    for doc_id, doc_title in docs:
        if doc_id not in seen_paper_idxs:
            seen_paper_idxs.add(doc_id)
            recommended_papers.append((doc_id, doc_title))
        if(len(recommended_papers)) == top_k:
            break

    return recommended_papers

# 특정 문서에 대하여 유사도 검색
def query_from_doc(
        query: str, 
        vectorstore: Chroma, 
        translator: Translator, 
        doc_id: int, 
        top_k: int 
    ) -> List[Tuple[str, str, str]]:
    
    translated_query = translator.translate(text=query, src="ko", dest="en")
    if isinstance(translated_query, Translated):
        query = translated_query.text

    results = vectorstore.similarity_search_with_score(query, k=top_k, filter={"doc_id": f"{doc_id}"})
    docs = [(result.metadata["doc_id"], result.metadata["doc_title"], result.page_content) for result, _ in results]

    return docs

############## 문서 검색 ##############


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
You are an AI assistant providing helpful advice.
You are given the following extracted parts of a long document and a question. Provide a conversational answer based on the context provided.
If you can't find the answer in the context below, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

Question: {question}
=========
{context}
=========
Answer in Markdown:''', input_variables=["question", "context"])

def make_chain(vectorstore: Chroma, llm, doc_id): 
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
                    "doc_id": doc_id
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
