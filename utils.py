import os
from typing import List
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever

class CustomVectorStoreRetriever(VectorStoreRetriever):
    vectorstore: Chroma
    def get_relevant_documents(self, query: str) -> List[Document]:
        if self.search_type == "similarity":
            results = self.vectorstore.similarity_search_with_score(query, **self.search_kwargs)
            docs = [result[0] for result in results]
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")

        docs = [Document(page_content=doc.metadata["origin_content"], metadata=doc.metadata) for doc in docs]
        docs = sorted(docs, key=lambda doc:doc.metadata["chunk_idx"])

        return docs

def load_pdf_files(path):
    files = os.listdir(path)
    pdf_files = [file for file in files if file.endswith('.pdf')]
    return pdf_files

def create_chunks(sentences, max_chunk_length=1000):
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Add the current sentence to the current chunk
        new_chunk = current_chunk + " " + sentence.strip()
        
        # Check if the new chunk length is within the limit
        if len(new_chunk) <= max_chunk_length:
            current_chunk = new_chunk
        else:
            # If the new chunk exceeds the limit, add the current chunk to the list of chunks
            chunks.append(current_chunk.strip())
            # Start a new chunk with the current sentence
            current_chunk = sentence.strip()
    
    current = current_chunk.strip()
    # Add the last chunk if it's not empty
    if current:
        if len(current) < 300:
            chunks[-1] = chunks[-1] + " " + current
        else:
            chunks.append(current)

    return chunks
