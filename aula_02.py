from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import TypedDict
import getpass
import os

if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Openai: ")

llm = ChatGroq(
    model = "openai/gpt-oss-120b"
)

## coleta de páginas
topicos = ["Inteligência Artificial", "História da Internet", "LangChain"]

docs = []

## split em chunks
for t in topicos:
    loader = WikipediaLoader(query=t, lang="pt", load_max_docs=1)
    docs.extend(loader.load())

tex_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    add_start_index=True
)

chunks = tex_splitter.split_documents(docs)


## Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1"
)

## banco de dados vetorial

persist_directory="./chroma_wiki"

if os.path.exists(persist_directory):
    print("Carregando banco vetorial existente...")
    vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_name="wikipedia_pt_rag"  
    )
else:
    print("Criando banco vetorial...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="wikipedia_pt_rag",
        persist_directory=persist_directory 
    )
    
##vectordb.persist()

## Retriever
retriever = vectordb.as_retriever(search_kwargs={"k":4})


def format_docs(docs_encontrados):
    contexto = "\n\n".join([
    f"{doc.page_content}\n[Fonte: {doc.metadata.get('title')}]"
    for doc in docs_encontrados]
)
    return contexto

prompt = ChatPromptTemplate.from_template("""
                                          
"Você é um assitente factual. Use EXCLUSIVAMENTE o contexto para respoder.\n"
"se não houver informação sufuciente, diga isso explicitamente. \n\n"                                                                                                                         
"Contexto: {contexto}\n\n"
"Pergunta: {pergunta}\n\n"                                                                      
As Repostas devem conter a fonte no final no formato, [fonte: título]                                      
""")

class RAGState(TypedDict, total=False):
    pergunta: str
    docs: list
    contexto: str
    resposta: str

def node_retrieve(state: RAGState) -> RAGState:
    docs = retriever.invoke(state["pergunta"])
    return {"docs": docs, "contexto": format_docs(docs)}

def node_augment(state:RAGState) -> RAGState:
    return {"contexto": state["contexto"]}
    
def node_generate(state: RAGState) -> RAGState:
    msg = prompt.format_messages(
            pergunta=state["pergunta"], contexto=state["contexto"]
        )
    out = llm.invoke(msg).content
    return {"resposta": out}

#criado o grafo
workflow = StateGraph(RAGState)

workflow.add_node("Retrieve", node_retrieve)
workflow.add_node("AugmentPrompt", node_augment)
workflow.add_node("Generate", node_generate)

workflow.set_entry_point("Retrieve")
workflow.add_edge("Retrieve","AugmentPrompt")
workflow.add_edge("AugmentPrompt", "Generate")
workflow.add_edge("Generate", END)

app = workflow.compile()

resultado = app.invoke({"pergunta": "O que é LangChain"})
print("resposta", resultado["resposta"])
print("fonte:", resultado["docs"][0].metadata.get("source"))