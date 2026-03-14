from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


## coleta de páginas
def load_documents():

    topicos = ["Inteligência Artificial", "História da Internet", "LangChain"]
    docs = []

    ## split em chunks
    for t in topicos:
        loader = WikipediaLoader(query=t, lang="pt", load_max_docs=1)
        docs.extend(loader.load())

    tex_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        add_start_index=True
    )

    chunks = tex_splitter.split_documents(docs)

    return chunks