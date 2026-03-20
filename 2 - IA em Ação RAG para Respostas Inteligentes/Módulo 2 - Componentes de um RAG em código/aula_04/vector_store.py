from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from loader import load_documents
import os

persist_directory="./chroma_pdf_html"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def get_vectorstore():
    if os.path.exists(persist_directory):
        print("Carregando banco vetorial existente...")
        vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="pdf_html_rag"  
        )
            
    else:
        chunks = load_documents()
        print("Criando banco vetorial...")
        vectordb = Chroma.from_documents(
        documents= chunks,
        embedding=embeddings,
        collection_name="pdf_html_rag",
        persist_directory=persist_directory 
        )

    return vectordb
        
    #retriever = vectordb.as_retriever(search_kwargs={"k":3})

    #consulta = "who is Alice?"

    #resultado = vectordb.similarity_search(consulta, k=3)

    #print("Quantidade de resultados:", len(resultado))
    #for d in resultado:
    #    print(d.metadata, d.page_content[:100], "\n---")

   
        
