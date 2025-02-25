import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# Import FAISS instead of Chroma
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

st.title("RAG Application built on Gemini Model")

# Add error handling for file loading
try:
    loader = PyPDFLoader("iag_pds.pdf")
    data = loader.load()
    st.success(f"Successfully loaded PDF with {len(data)} pages")
except Exception as e:
    st.error(f"Error loading PDF: {e}")
    st.stop()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)
st.success(f"Document split into {len(docs)} chunks")

# Create embeddings
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create FAISS vector store instead of Chroma
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )
    st.success("Vector store created successfully")
except Exception as e:
    st.error(f"Error creating vector store: {e}")
    st.stop()

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

# Create chat interface
query = st.chat_input("Ask a question about the document: ")

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

if query:
    with st.spinner("Searching document and generating answer..."):
        try:
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            response = rag_chain.invoke({"input": query})
            st.write(response["answer"])
        except Exception as e:
            st.error(f"Error generating response: {e}")