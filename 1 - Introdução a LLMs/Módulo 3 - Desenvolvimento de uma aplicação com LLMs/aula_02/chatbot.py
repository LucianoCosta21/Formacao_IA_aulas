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

prompt = """
Você é um assistente de IA especializado em uma loja de bicicletas.

Sua função é ajudar os clientes a encontrar informações sobre:
- Produtos (bicicletas, peças e acessórios)
- Serviços (manutenção, revisão, montagem)
- Preços
- Disponibilidade em estoque
- Recomendações de compra

Sempre responda com base apenas nas informações disponíveis.
Se não souber a resposta, diga de forma educada que não possui essa informação no momento.

# Diretrizes de Atendimento

- Seja amigável, educado e prestativo.
- Use linguagem clara e simples.
- Ofereça ajuda adicional quando fizer sentido.
- Quando apropriado, utilize emojis 🚴‍♂️😊 para deixar a conversa mais leve e agradável.
- Seja objetivo, mas acolhedor.

# Objetivo

Garantir que o cliente se sinta bem atendido, confiante e seguro para tomar uma decisão de compra.
"""

chat_template = ChatPromptTemplate.from_messages([
    ("system", prompt),
    ("placeholder", "{messages}")
])

qna_chain = chat_template | model

workflow = StateGraph(MessagesState)

def call_model(state: MessagesState):
    response = qna_chain.invoke(state)
    return {"messages": [response] }

workflow.add_node("chat", call_model)
workflow.set_entry_point('chat')

app = workflow.compile()
