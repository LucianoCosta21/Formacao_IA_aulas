from graph import build_graph

app = build_graph()

resultado = app.invoke({"pergunta": "O que é LangChain"})

print("resposta", resultado["resposta"])
print("fonte:", resultado["docs"][0].metadata.get("source"))