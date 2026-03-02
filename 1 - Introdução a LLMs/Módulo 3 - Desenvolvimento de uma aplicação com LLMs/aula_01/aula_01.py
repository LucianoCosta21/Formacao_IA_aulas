import streamlit as st
from langchain_groq import ChatGroq
import getpass
import os


st.title("Chat com IA usando LangChain")
prompt = st.text_area("Digite sua pergunta:")

if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Openai: ")

if st.button("Enviar"):
    model = ChatGroq(
        model = "openai/gpt-oss-120b",
    )
    resposta = model.invoke(prompt)
    st.write("**Resposta:**", resposta.content)
else: 
    st.warning("Digite uma pergunta antes de enviar")