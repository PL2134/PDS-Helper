import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

# Initialize session state to store chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("RAG Chatbot for IAG PDS")

# Function to initialize RAG components (vectorstore, retriever, llm)
@st.cache_resource
def initialize_rag():
    try:
        # Load and process the PDF
        loader = PyPDFLoader("iag_pds.pdf")
        data = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        docs = text_splitter.split_documents(data)
        
        # Create embeddings and vectorstore
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
        
        # Create retriever
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2, max_tokens=None, timeout=None)
        
        return retriever, llm
    except Exception as e:
        st.error(f"Error initializing RAG components: {e}")
        return None, None

# Initialize RAG components
retriever, llm = initialize_rag()

# If initialization was successful
if retriever and llm:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
    # If this is a new conversation, show a welcome message
    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            st.write("How can I help you? Ask me anything about the document.")
    
    # Get user input
    user_query = st.chat_input("Ask a question about the document:")
    
    if user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # System prompt for RAG
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
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating answer..."):
                try:
                    question_answer_chain = create_stuff_documents_chain(llm, prompt)
                    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                    
                    response = rag_chain.invoke({"input": user_query})
                    assistant_response = response["answer"]
                    
                    # Display assistant response
                    st.write(assistant_response)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                except Exception as e:
                    error_message = f"Error generating response: {e}"
                    st.error(error_message)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message})
else:
    st.error("Failed to initialize the application. Please check if the PDF file exists and API keys are set correctly.")