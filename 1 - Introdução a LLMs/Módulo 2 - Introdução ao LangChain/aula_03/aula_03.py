from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import getpass
import os
import pdf_analise 

class State(dict):
    pergunta: str
    contexto: list
    resposta: str

def generate(state: State):
    docs_content = state["contexto"]
    response = qna_chain.invoke({
        "pergunta": state["pergunta"],
        "contexto": docs_content
        }) 
    return {"resposta": response.content}

template = """
Responda a pergunta abaixo com base apenas no contexto fornecido.

Contexto:
{contexto}

Pergunta:
{pergunta}

Se a resposta não estiver no contexto, diga:
"Não encontrei essa informação no documento."
"""

if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Openai: ")

model = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature=0.8)

prompt = ChatPromptTemplate.from_template(template)

qna_chain = prompt | model

estado = {
    "pergunta": "Lista dos governadores do Rio de Janeiro de 1975 - 2005",
    "contexto": pdf_analise.text
}

resultado = generate(estado)

print(resultado["resposta"])