from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("aula_03/Historico_Estado.pdf") 
docs = loader.load()

text = "\n\n".join([doc.page_content for doc in docs])

##print(text)
