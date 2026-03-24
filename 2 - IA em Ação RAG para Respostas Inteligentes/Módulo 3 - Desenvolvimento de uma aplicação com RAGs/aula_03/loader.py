from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents():
    topicos = "Hunter x Hunter"
    docs = []

    loader = WikipediaLoader(query=topicos, lang="pt", load_max_docs=1)
    docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(docs)
    
    return chunks
