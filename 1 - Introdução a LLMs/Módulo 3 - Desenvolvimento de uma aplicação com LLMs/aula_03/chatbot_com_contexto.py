from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, MessagesState, StateGraph
import getpass
import os

if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Openai: ")

model = ChatGroq(
    model = "openai/gpt-oss-120b"
    )

template = """
Responda a pergunta abaixo com base apenas no contexto fornecido.

Contexto:
{contexto}

Pergunta:
{pergunta}

Se a resposta não estiver no contexto, diga:
"Não encontrei essa informação no documento."
"""

prompt = ChatPromptTemplate.from_template(template)

qna_chain = prompt | model

class State(dict):
    pergunta: str
    contexto: list
    resposta: str

workflow = StateGraph(State)

def generate(state: State):
    docs_content = state["contexto"]
    response = qna_chain.invoke({
        "pergunta": state["pergunta"],
        "contexto": docs_content
    }) 
    return {"resposta": response.content}


workflow.add_node("generate", generate)
workflow.set_entry_point("generate")
workflow.set_finish_point("generate")

app = workflow.compile()
