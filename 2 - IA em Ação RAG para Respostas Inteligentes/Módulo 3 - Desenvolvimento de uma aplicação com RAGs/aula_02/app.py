import streamlit as st
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage, AIMessage
import os
import getpass

if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Openai: ")

if"chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
st.title("Chatbot com Memória - Aula 2")
user_input = st.text_area("Digite sua pergunta:")
interacoes = [msg.content for msg in st.session_state.chat_history if isinstance(msg, HumanMessage)]
st.subheader(f"Interações: {len(interacoes)}", divider=True)

send = st.button("Enviar")
limpar = st.button("Limpar histórico")

modelo = ChatGroq(model = "openai/gpt-oss-120b")

if send and user_input:
    
    st.session_state.chat_history.append(HumanMessage(content=user_input))
   
    response = modelo.invoke(st.session_state.chat_history)

    st.session_state.chat_history.append(AIMessage(content= response.content))
    
    st.write(response.content)
   

if limpar:
    st.session_state.chat_history = []
    st.rerun()

