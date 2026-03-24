from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os


persist_directory="./chroma_wiki"
print(os.path.exists(persist_directory))

embeddings = HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1"
    )

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_name="wikipedia_pt_rag"  
)
##Busca simples
"""
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

query = "Quais foram os principais marcos da Inteligência Artificial?"

resultado = retriever.invoke(query)
for doc in resultado:
    print(doc.page_content[:200])
    print(doc.metadata)
    print("-" * 50)
"""
##Teste o parâmetro top_k

for k in [1,3,5]:
    
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    
    query = "Quais foram os principais marcos da Inteligência Artificial?"
    
    resultado = retriever.invoke(query)
    
    for doc in resultado:
        print(doc.page_content[:200])



