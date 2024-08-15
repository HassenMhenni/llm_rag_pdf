from langchain_community.vectorstores.chroma import Chroma
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata

# Global variables
vector_store = None
retriever = None
chain = None

def create_model():
    return ChatOllama(model="llama3:8b")

def create_text_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

def create_prompt():
    return PromptTemplate.from_template(
        """
        [INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.<</SYS>> 
        Question: {question} 
        Context: {context} 
        Answer: [/INST]
        """
    )

def ingest(pdf_path):
    global vector_store, retriever, chain
    
    model = create_model()
    text_splitter = create_text_splitter()
    prompt = create_prompt()
    
    docs = PyPDFLoader(file_path=pdf_path).load()
    chunks = text_splitter.split_documents(docs)
    chunks = filter_complex_metadata(chunks)

    vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            'k': 3,
            'score_threshold': 0.5
        },
    )

    chain = ({
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
    )

def ask(query: str):
    if not chain:
        return "Please ingest a PDF file first."
    return chain.invoke(query)

def clear():
    global vector_store, retriever, chain
    vector_store = None
    retriever = None
    chain = None