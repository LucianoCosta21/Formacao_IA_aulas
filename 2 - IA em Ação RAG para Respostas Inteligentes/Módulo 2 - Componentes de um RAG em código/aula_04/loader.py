from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
def load_documents():
    loader_pdf = PyPDFLoader(file_path= "Attention_Is_All_You_Need.pdf")
    documento_pdf = loader_pdf.load()

    loader_html = BSHTMLLoader(
        file_path= "Alices Adventures in Wonderland _ Project Gutenberg.html",
        open_encoding="utf-8",
        bs_kwargs={"features": "html.parser"}
    )
    documento_html = loader_html.load()
    documentos = documento_pdf + documento_html
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documentos)
    return chunks

##print(documento_pdf[0].page_content[:500])
##print(documento_html[0].metadata)