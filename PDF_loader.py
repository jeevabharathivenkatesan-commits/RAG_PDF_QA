import os  # directory path
from dotenv import load_dotenv # from .env file
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader  #alternative for UnstructuredPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# it takes absolute path of current pdf_loader script file location as working_dir
working_dir = os.path.dirname(os.path.abspath(__file__))

#initialize the embeddings model
embeddings = HuggingFaceEmbeddings()

#LLM
llm=ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3
)

def process_document(file_name):
    # finds the uploaded pdf file
    loader=PyPDFLoader(f"{working_dir}/{file_name}")
    documents=loader.load()

    #Chunking
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500
    )

    texts=text_splitter.split_documents(documents)

    #To vector chroma
    vectordb=Chroma.from_documents(
        embedding=embeddings,
        documents=texts,
        persist_directory=f"{working_dir}/vectorstore"
    )
    return 0

def rag(user_query):
    # to access vector store
    vectordb=Chroma(
        persist_directory=f"{working_dir}/vectorstore",
        embedding_function=embeddings
    )
    # initialize retriever
    retriever=vectordb.as_retriever()

    #
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    response=qa_chain.invoke({"query": user_query})
    answer=response["result"]

    return answer

