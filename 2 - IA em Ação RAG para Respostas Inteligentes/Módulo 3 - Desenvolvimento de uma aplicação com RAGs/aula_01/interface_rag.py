import streamlit as st
from langchain_groq import ChatGroq
import os
import getpass

if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Openai: ")

st.title("💬 Meu primeiro sistema RAG")
st.subheader("Sistema de IA com LangChain + OpenAI")
pergunta = st.text_area("Digite sua pergunta:")

if st.button("Enviar"):
    with st.spinner("Carregando resposta...", show_time=False):
        modelo = ChatGroq(model = "openai/gpt-oss-120b"
        )
        resposta = modelo.invoke(pergunta)
        st.write("### Respsota:")
        st.write(resposta.content)
   
   




