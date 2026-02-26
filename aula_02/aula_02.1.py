#from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.messages import HumanMessage, SystemMessage
#from langchain_core.prompts import ChatPromptTemplate
#import langchain
import getpass
import os

if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Openai: ")

model = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature=0)

system_template = "Traduza o seguinte texto de Inglês para {idioma}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), "user", "{text}"]
)

texto = input("Digite a palavra que deseja traduzir... ")

prompt = prompt_template.invoke({"idioma": "Português", "text": f"{texto}"})
response = model.invoke(prompt)

#response = model.invoke([
#    SystemMessage("Traduza o seguinte texto de Inglês para português"),
#    HumanMessage("Hi!")
#])

print(response.content)

 