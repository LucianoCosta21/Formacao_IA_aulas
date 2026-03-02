import streamlit as st
from chatbot import app
from langchain_core.messages import AIMessage, HumanMessage


st.set_page_config(layout='wide', page_title='Chatbot de loja de bicicletas', page_icon='oTo')

st.title("Loja de Bicicletas - Assistente Vitural")

if 'message_history' not in st.session_state:
    st.session_state.message_history = [AIMessage(content="Olá, sou um assistente vitural para ajudar você a encontrar informações sobre produtos de uma loja de bicileta. Como posso ajudar?")]

user_input = st.chat_input("Digite aqui...")

if user_input:
    st.session_state.message_history.append(HumanMessage(content=user_input))
    response = app.invoke({
        'messages': st.session_state.message_history
    })

    st.session_state.message_history = response['messages']

for message in st.session_state.message_history:
    
    if isinstance(message, AIMessage):
        message_box = st.chat_message('assistant')
    else:
        message_box = st.chat_message('user')

    message_box.markdown(message.content)