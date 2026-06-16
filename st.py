import streamlit as st
from doc_ingest import process_pdf, ask

st.title("PDF RAG Chatbot")

pdf = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if pdf:

    with open(pdf.name, "wb") as f:
        f.write(pdf.getbuffer())

    vectordb = process_pdf(pdf.name)

    query = st.text_input("Ask Question")

    if st.button("Get Answer"):

        answer = ask(vectordb, query)

        st.write(answer)
