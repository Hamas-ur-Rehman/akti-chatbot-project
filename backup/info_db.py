import os
import bs4
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import WebBaseLoader

def extract_combined_text(pages):
    combined_text = ""
    for page in pages:
        combined_text += page.page_content + "\n"
    return combined_text

def pdf_database(question):
    file_path = "files/pdf/"
    pdf_files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.pdf')]

    for i in pdf_files:
        loader = PyPDFLoader(i)
        pages = []
        for page in loader.load():
            pages.append(page)
        
    vector_store = InMemoryVectorStore.from_documents(pages, OpenAIEmbeddings())

    documents = vector_store.similarity_search(question)  
    return extract_combined_text(documents)  

def csv_database(question):
    file_path = "files/csv/"
    csv_files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.csv')]

    for i in csv_files:
        loader = CSVLoader(i)
        pages = []
        for page in loader.load():
            pages.append(page)
        
    vector_store = InMemoryVectorStore.from_documents(pages, OpenAIEmbeddings())

    documents = vector_store.similarity_search(question)  
    return extract_combined_text(documents)

def web_database(question):
    loader = WebBaseLoader()
    pages = [
        "https://www.dawn.com/",
        "https://www.bbc.com/"
    ]
    docs = []
    for page in pages:
        loader = WebBaseLoader(web_paths=[page])
        for doc in loader.load():
            docs.append(doc)
    
    vector_store = InMemoryVectorStore.from_documents(docs, OpenAIEmbeddings())

    documents = vector_store.similarity_search(question)  
    return extract_combined_text(documents)

