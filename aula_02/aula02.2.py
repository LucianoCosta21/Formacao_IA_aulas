import uuid
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import getpass
import os

if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Openai: ")

model = ChatGroq(
    model = "openai/gpt-oss-120b")

workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id" : thread_id}}

query = "Olá, eu sou o Luciano!"
input_messages = (["user", query ])

output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

query = "como me chamo?"
input_messages = (["user", query ])

output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()



 