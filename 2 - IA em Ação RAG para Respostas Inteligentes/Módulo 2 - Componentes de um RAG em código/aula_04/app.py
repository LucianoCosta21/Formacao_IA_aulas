from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from vector_store import get_vectorstore
import getpass
import os



vectordb = get_vectorstore()
retriever = vectordb.as_retriever(search_kwargs={"k":3})

if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Openai: ")

llm = ChatGroq(
    model = "openai/gpt-oss-120b"
)

resposta = llm.invoke("who is Alice?")

#print(resposta.content)

#--------------------------------------------------------
def format_docs(docs_encontrados):
    contexto = "\n\n".join([
    f"{doc.page_content}\n[Fonte: {doc.metadata.get('title')}]"
    for doc in docs_encontrados]
)
    return contexto

pergunta = "who is Alice?"
docs = retriever.invoke(pergunta)
print(len(docs))
contexto = format_docs(docs)

prompt = ChatPromptTemplate.from_template("""
Você é um assistente factual.

Use APENAS o contexto abaixo para responder.
Se não souber, diga que não há informação suficiente.

Contexto:
{contexto}

Pergunta:
{pergunta}

Responda apenas com base nas informações fornecidas e cite as fontes no final.
""")

msg = prompt.format_messages(
    contexto=contexto,
    pergunta=pergunta
    
)

resposta_com_contexto = llm.invoke(msg)

print(resposta_com_contexto.content)