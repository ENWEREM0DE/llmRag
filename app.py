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

with st.sidebar:
    st.title("🤖 Chat with PDF 📒")
    load_dotenv()
    st.markdown("""
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [Langchain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models)
    """)
    add_vertical_space(5)
    st.write("Made with ❤️ by [Don](https://github.com/ENWEREM0DE)")

def main():
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
            # Only create embeddings object with empty client to load existing index
            embeddings = OpenAIEmbeddings(openai_api_key="")
            VectorStore = FAISS.load_local(f"{store_name}.pkl", embeddings, allow_dangerous_deserialization=True)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            VectorStore.save_local(f"{store_name}.pkl")

        
        st.markdown('<div class="chat-section">', unsafe_allow_html=True)
        st.subheader("💭 Ask Questions About Your Document")
        query = st.text_input("Type your question here:", placeholder="e.g., What are the main topics discussed in this document?")

        if query:
            with st.spinner('Thinking...'):
                docs = VectorStore.similarity_search(query=query, k=3)
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=query)
                
                st.markdown("### Answer:")
                st.markdown(response)
        

        
        
if __name__ == "__main__":
    main()