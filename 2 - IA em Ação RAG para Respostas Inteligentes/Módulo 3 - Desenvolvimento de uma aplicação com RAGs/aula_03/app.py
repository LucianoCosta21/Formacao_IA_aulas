import streamlit as st
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage, AIMessage
from vector_store import get_vectorstore
from langchain_core.prompts import ChatPromptTemplate
import os
import getpass

if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Openai: ")

if"chat_history" not in st.session_state:
    st.session_state.chat_history = []

vectordb = get_vectorstore()



score_threshold = st.slider("Limite mínimo de similaridade (score_threshold):", 0.0, 1.0, 0.5)

top_k = st.slider("Número de documentos a recuperar (top_k):", 1, 10, 3)

retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": top_k,
        "score_threshold": score_threshold
    }
    )

st.title("Chatbot com Memória - Aula 3")


user_input = st.text_area("Digite sua pergunta:")


interacoes = [msg.content for msg in st.session_state.chat_history if isinstance(msg, HumanMessage)]
st.subheader(f"Interações: {len(interacoes)}", divider=True)


send = st.button("Enviar")
limpar = st.button("Limpar histórico")

modelo = ChatGroq(model = "openai/gpt-oss-120b")


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

if send and user_input:

    docs = retriever.invoke(user_input)

    contexto = "\n\n".join([
    f"{doc.page_content}\n[Fonte: {doc.metadata.get('title')}]"
    for doc in docs]
    )
    
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    msg = prompt.format_messages(
        contexto=contexto,
        pergunta= user_input
    )

    response = modelo.invoke(msg)

    st.session_state.chat_history.append(AIMessage(content= response.content))

    st.write(response.content)
   

if limpar:
    st.session_state.chat_history = []
    st.rerun()

