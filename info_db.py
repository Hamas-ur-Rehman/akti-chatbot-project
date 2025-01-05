import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import WebBaseLoader

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

def extract_combined_text(pages):
    return " ".join([page.page_content for page in pages])

def pdf_database(question):
    file_path = "files/pdf/"
    pdf_files = [os.path.join(file_path,f) for f in os.listdir(file_path) if f.endswith('.pdf')]

    pages = []
    for i in pdf_files:
        loader = PyPDFLoader(i)
        for page in loader.load():
            pages.append(page)

    vector_store = InMemoryVectorStore.from_documents(pages,OpenAIEmbeddings())
    documents = vector_store.similarity_search(question)
    return extract_combined_text(documents)

def csv_database(question):
    file_path = "files/csv/"
    csv_files = [os.path.join(file_path,f) for f in os.listdir(file_path) if f.endswith('.csv')]

    pages = []
    for i in csv_files:
        loader = CSVLoader(i)
        for page in loader.load():
            pages.append(page)

    vector_store = InMemoryVectorStore.from_documents(pages,OpenAIEmbeddings())
    documents = vector_store.similarity_search(question)
    return extract_combined_text(documents)

def web_database(question):
    pages = [
        "https://www.dawn.com/",
        "https://www.bbc.com/"
    ]
    docs = []

    for data in pages:
        loader = WebBaseLoader(data)
        for doc in loader.load():
            docs.append(doc)
        
    vector_store = InMemoryVectorStore.from_documents(docs,OpenAIEmbeddings())
    documents = vector_store.similarity_search(question)
    return extract_combined_text(documents)