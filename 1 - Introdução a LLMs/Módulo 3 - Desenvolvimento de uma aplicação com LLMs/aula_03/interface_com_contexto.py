import streamlit as st
from chatbot_com_contexto import app
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader


st.set_page_config(layout='wide', page_title= 'Chatbot História da Arquitetura e Urbanismo', page_icon='oTo')

st.title("História da Arquitetura e Urbanismo - Assistente Vitural")

if 'message_history' not in st.session_state:
    st.session_state.message_history = [AIMessage(content="Olá, sou um assistente vitural para ajudar você a aprender sobre a história da arquitetura. Como posso ajudar?")]

uploaded_pdf = st.file_uploader("Faça seu upload de um PDF para análise", type=["pdf"])

pdf_text = "" 

if uploaded_pdf is not None:
    file_name = uploaded_pdf.name
    pdf_loader = PyPDFLoader(file_name)
    docs = pdf_loader.load()
    pdf_text = "\n\n".join([doc.page_content for doc in docs][:10])

""

user_input = st.chat_input("Digite aqui...")

if user_input:
    st.session_state.message_history.append(HumanMessage(content=user_input))
    response = app.invoke({
        'pergunta': user_input,
        "contexto": [pdf_text ]
    })
    st.session_state.message_history.append(
        AIMessage(content=response["resposta"]))

for message in st.session_state.message_history:
    
    if isinstance(message, AIMessage):
        message_box = st.chat_message('assistant')
    else:
        message_box = st.chat_message('user')

    message_box.markdown(message.content)