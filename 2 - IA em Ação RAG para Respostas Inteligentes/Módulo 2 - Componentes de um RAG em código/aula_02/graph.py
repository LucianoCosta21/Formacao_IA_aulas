from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from vector_store import get_vectorstore
from typing import TypedDict
import getpass
import os

vectordb = get_vectorstore()
retriever = vectordb.as_retriever(search_kwargs={"k":4})

if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Openai: ")

llm = ChatGroq(
    model = "openai/gpt-oss-120b"
)


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
def build_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("Retrieve", node_retrieve)
    workflow.add_node("AugmentPrompt", node_augment)
    workflow.add_node("Generate", node_generate)

    workflow.set_entry_point("Retrieve")
    workflow.add_edge("Retrieve","AugmentPrompt")
    workflow.add_edge("AugmentPrompt", "Generate")
    workflow.add_edge("Generate", END)

    app = workflow.compile()
    return app