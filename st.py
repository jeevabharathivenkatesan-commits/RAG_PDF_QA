import os
import streamlit as st
from PDF_loader import rag,process_document

st.title("RAG- Upload a pdf and get answer for your query")
st.set_page_config(
    page_title="QndA",
    page_icon="🗺️",
    layout="centered" # alternative wide
)

working_dir = os.path.dirname(os.path.abspath(__file__))

uploaded_file=st.file_uploader("Upload a file",type=["pdf","xlsx"])

if uploaded_file is not None:
    save_path=os.path.join(working_dir,uploaded_file.name)
    #save the file
    with open(save_path,mode='wb') as f:
        f.write(uploaded_file.getbuffer())

    #processing document
    process_document=process_document(uploaded_file.name)
    st.info("Processed Successfully")


user_query=st.text_input("Your query?")
if st.button("Get answer"):
    answer=rag(user_query)
    st.markdown("Response")
    st.markdown(answer)