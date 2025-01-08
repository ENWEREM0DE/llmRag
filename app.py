import os
from dotenv import load_dotenv
import streamlit as st 
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI

# Load env variables at the start
load_dotenv()
DONS_SPECIAL_PASSWORD = os.getenv('DONS_SPECIAL_PASSWORD')

def check_authentication():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.api_key = None

    if not st.session_state.authenticated:
        st.title("🔐 Authentication Required")
        auth_option = st.radio("Choose authentication method:", 
                             ["Use Don's Special Password", "Use Your Own OpenAI API Key"])
        
        if auth_option == "Use Don's Special Password":
            password = st.text_input("Enter Don's Special Password:", type="password")
            if st.button("Submit Password"):
                if password == DONS_SPECIAL_PASSWORD:
                    st.session_state.authenticated = True
                    st.session_state.api_key = os.getenv('OPENAI_API_KEY')
                    st.success("Authentication successful!")
                    st.rerun()
                else:
                    st.error("Invalid password!")
        else:
            api_key = st.text_input("Enter your OpenAI API Key:", type="password")
            if st.button("Submit API Key"):
                if api_key.startswith('sk-'):  # Basic validation
                    st.session_state.authenticated = True
                    st.session_state.api_key = api_key
                    st.success("API Key accepted!")
                    st.rerun()
                else:
                    st.error("Invalid API Key format!")
        
        return False
    return True

def main():
    # Move sidebar outside of authentication check
    with st.sidebar:
        st.title("🤖 Chat with PDF 📒")
        # ... rest of sidebar code ...

    # Check authentication before showing main content
    if not check_authentication():
        return

    # Main content (only shown after authentication)
    st.header("Chat with PDF 💬")

    
    # Upload Section
    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📄 Upload Your PDF Document")
        pdf = st.file_uploader("Drag and drop your PDF here", type="pdf")
        if pdf is not None:
            st.markdown(f'<div class="file-info">📎 Current file: {pdf.name}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.api_key)
            VectorStore = FAISS.load_local(f"{store_name}.pkl", embeddings, allow_dangerous_deserialization=True)
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.api_key)
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            VectorStore.save_local(f"{store_name}.pkl")

        
        st.markdown('<div class="chat-section">', unsafe_allow_html=True)
        st.subheader("💭 Ask Questions About Your Document")
        query = st.text_input("Type your question here:", placeholder="e.g., What are the main topics discussed in this document?")

        if query:
            with st.spinner('Thinking...'):
                docs = VectorStore.similarity_search(query=query, k=3)
                llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo", 
                    temperature=0,
                    openai_api_key=st.session_state.api_key
                )
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=query)
                
                st.markdown("### Answer:")
                st.markdown(response)
        

        
        
if __name__ == "__main__":
    main()