from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from loader import load_documents
import os


persist_directory="./chroma_wiki"

def get_vectorstore():

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(persist_directory):
        print("Carregando banco vetorial existente...")
        vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="wikipedia_pt_rag"  
    )
        
    else:
        chunks = load_documents()
        print("Criando banco vetorial...")
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="wikipedia_pt_rag",
            persist_directory=persist_directory 
    )
        
    return vectordb
     
    