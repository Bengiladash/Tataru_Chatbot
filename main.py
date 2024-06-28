import streamlit as st
import os
import pickle
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(layout="wide")

# Ensure necessary directories and configurations
DOCS_DIR = os.path.abspath("./uploaded_docs")
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)

vector_store_path = "vectorstore.pkl"

# Sidebar: Document Upload Section
with st.sidebar:
    st.subheader("Add to the Knowledge Base")
    with st.form("my-form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Upload a file to the Knowledge Base:", accept_multiple_files=True)
        submitted = st.form_submit_button("Upload!")

    if uploaded_files and submitted:
        for uploaded_file in uploaded_files:
            st.success(f"File {uploaded_file.name} uploaded successfully!")
            with open(os.path.join(DOCS_DIR, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.read())

# Load raw documents
raw_documents = DirectoryLoader(DOCS_DIR).load()

# Debugging: Print the loaded documents
print("Loaded documents:", raw_documents)

# Option for using an existing vector store
with st.sidebar:
    use_existing_vector_store = st.radio("Use existing vector store if available", ["Yes", "No"], horizontal=True)

# Initialize vectorstore
vectorstore = None
if use_existing_vector_store == "Yes" and os.path.exists(vector_store_path):
    with open(vector_store_path, "rb") as f:
        vectorstore = pickle.load(f)
    st.sidebar.success("Existing vector store loaded successfully.")
else:
    if raw_documents:
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        documents = text_splitter.split_documents(raw_documents)

        # Debugging: Print the split documents
        print("Split documents:", documents)

        try:
            embeddings = NVIDIAEmbeddings(model="NV-Embed-QA", embed_documents="passage")
            vectorstore = FAISS.from_documents(documents, embeddings)
            with open(vector_store_path, "wb") as f:
                pickle.dump(vectorstore, f)
            st.sidebar.success("Vector store created and saved.")
        except Exception as e:
            # Debugging: Print the exception
            print("Exception during vector store creation:", e)
            st.sidebar.error(f"Error creating vector store: {e}")
    else:
        st.sidebar.warning("No documents available to process!", icon="⚠️")

# Chat Interface
st.subheader("Chat with your AI Assistant, Tataru!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt_template = ChatPromptTemplate.from_messages(
    [("system",
      "You are a helpful AI assistant who works for CSUSB, your purpose is to provide information from CSUSB's knowledge database, your name is Tataru are you have a cheery yet proffesional personality. You will reply to questions only based on the context that you are provided. If something is out of context, you will refrain from replying and politely decline to respond to the user. You will provide any relevant hyperlinks corresponding to any information you give if one exists, the hyperlink can only come from the custom data given to you. You will also provide a hyperlink to where you obtained your information from."),
     ("user", "{input}")]
)

user_input = st.chat_input("Ask a question to Tataru:")
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
chain = prompt_template | llm | StrOutputParser()

if user_input and vectorstore is not None:
    st.session_state.messages.append({"role": "user", "content": user_input})
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    context = ""
    for doc in docs:
        context += doc.page_content + "\n\n"

    augmented_user_input = "Context: " + context + "\n\nQuestion: " + user_input + "\n"

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for response in chain.stream({"input": augmented_user_input}):
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Feedback buttons
    feedback_col = st.columns([1])[0]
    with feedback_col:
        if st.button("👍", key=f"thumbs_up_{len(st.session_state.messages)}"):
            st.session_state.messages[-1]["feedback"] = "positive"
            st.toast("Thanks for your feedback!", icon="👍")
        if st.button("👎", key=f"thumbs_down_{len(st.session_state.messages)}"):
            st.session_state.messages[-1]["feedback"] = "negative"
            st.toast("Sorry to hear that. We'll use your feedback to improve!", icon="👎")
