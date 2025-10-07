import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# --- SESSION STATE MANAGEMENT ---
if "store" not in st.session_state:
    st.session_state.store = {}
if "conversational_rag_chain" not in st.session_state:
    st.session_state.conversational_rag_chain = None

# --- CHAT HISTORY MANAGEMENT FUNCTION ---
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# --- MAIN APP ---
st.set_page_config(page_title="Chat with PDFs", page_icon="🤖")
st.title("Chat with Your Documents")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.title("Configuration")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get your free API key from https://console.groq.com/keys"
    )

# --- MAIN LOGIC ---
if not groq_api_key:
    st.info("Please enter your Groq API key in the sidebar to continue.")
else:
    # --- UI FOR FILE UPLOAD ---
    with st.sidebar:
        st.subheader("Your Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files", type="pdf", accept_multiple_files=True
        )

        if st.button("Process Documents"):
            if not uploaded_files:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing documents..."):
                    documents = []
                    for uploaded_file in uploaded_files:
                        temp_file_path = f"./temp_{uploaded_file.name}"
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        loader = PyPDFLoader(temp_file_path)
                        documents.extend(loader.load())
                        os.remove(temp_file_path)

                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    splits = text_splitter.split_documents(documents)
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
                    retriever = vector_store.as_retriever()
                    
                    # *** KEY CHANGE HERE: Pass the API key directly ***
                    llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-8b-8192", temperature=0)

                    contextualize_q_system_prompt = (
                        "Given a chat history and the latest user question..." # (rest of the prompt)
                    )
                    contextualize_q_prompt = ChatPromptTemplate.from_messages([
                        ("system", contextualize_q_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ])
                    qa_system_prompt = (
                        "You are an assistant for question-answering tasks..." # (rest of the prompt)
                    )
                    qa_prompt = ChatPromptTemplate.from_messages([
                        ("system", qa_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ])

                    retrieval_chain = create_stuff_documents_chain(llm, qa_prompt)
                    conversational_rag_chain = create_retrieval_chain(retriever, retrieval_chain)
                    
                    st.session_state.conversational_rag_chain = RunnableWithMessageHistory(
                        conversational_rag_chain,
                        get_session_history,
                        input_messages_key="input",
                        history_messages_key="chat_history",
                        output_messages_key="answer",
                    )
                    st.success("Documents processed! You can now ask questions.")

    # --- CHAT INTERFACE ---
    if st.session_state.conversational_rag_chain:
        session_id = "user_session_1"
        history = get_session_history(session_id)
        for msg in history.messages:
            with st.chat_message(msg.type):
                st.markdown(msg.content)

        if prompt := st.chat_input("Ask a question about your documents..."):
            with st.chat_message("human"):
                st.markdown(prompt)
            with st.spinner("Thinking..."):
                response = st.session_state.conversational_rag_chain.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": session_id}},
                )
                with st.chat_message("ai"):
                    st.markdown(response["answer"])
    else:
        st.info("Please upload and process your PDF documents to start chatting.")
