import os
import streamlit as st
import PyPDF2
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai.embeddings import OpenAIEmbeddings


# Streamlit UI elements
st.title("Chat and Summarize PDF Document")


# Initialize session state variables
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "pdf_file" not in st.session_state:
    st.session_state.pdf_file = None
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None

# Input field for OpenAI API key
st.session_state.api_key = st.text_input("Enter your OpenAI API key:", st.session_state.api_key, type="password")

# File uploader for PDF files
uploaded_file = st.file_uploader("Upload a PDF file:", type="pdf")
if uploaded_file is not None:
    st.session_state.pdf_file = uploaded_file

# Button to start the process
if st.button("Start Analysis"):
    if not st.session_state.pdf_file or not st.session_state.api_key:
        st.error("Please provide both a valid PDF file and OpenAI API key.")
    else:
        # Set the OpenAI API key in environment variable
        os.environ['OPENAI_API_KEY'] = st.session_state.api_key

        # Initialize the OpenAI model
        model = ChatOpenAI(openai_api_key=st.session_state.api_key, model="gpt-3.5-turbo")
        parser = StrOutputParser()
        
        # Prompt for summarization
        summary_template = """
        Summarize the following text and highlight the key points:

        Text: {text}
        """
        summary_prompt = ChatPromptTemplate.from_template(summary_template)

        # Prompt for Q&A
        qa_template = """
        Answer the question based on the context below. If you can't 
        answer the question, reply "I don't know".

        Context: {context}

        Question: {question}
        """
        qa_prompt = ChatPromptTemplate.from_template(qa_template)

        # Extract text from the PDF file
        pdf_reader = PyPDF2.PdfReader(st.session_state.pdf_file)
        extracted_text = ""
        for page in pdf_reader.pages:
            extracted_text += page.extract_text()

        st.session_state.extracted_text = extracted_text

        # Display extracted text
        st.text_area("Extracted Text from PDF:", st.session_state.extracted_text, height=300)

        # Generate a summary of the extracted text
        summary = model({"messages": [{"role": "system", "content": summary_prompt.format(text=st.session_state.extracted_text)}]})
        st.session_state.summary = summary["choices"][0]["message"]["content"]

        # Display the summary
        st.text_area("Summary of PDF Document:", st.session_state.summary, height=200)

        # Save extracted text to a file
        with open("extracted_text.txt", "w") as file:
            file.write(st.session_state.extracted_text)

        # Use TextLoader to load extracted text file
        loader = TextLoader("extracted_text.txt")
        text_documents = loader.load()

        # Split the text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        documents = text_splitter.split_documents(text_documents)

        # Initialize embeddings and vector store
        embeddings = OpenAIEmbeddings()
        st.session_state.vectorstore = DocArrayInMemorySearch.from_documents(documents, embeddings)

        # Build the retrieval-augmented generation (RAG) chain
        st.session_state.chain = (
            {"context": st.session_state.vectorstore.as_retriever(), "question": RunnablePassthrough()}
            | qa_prompt
            | model
            | parser
        )

# Check if transcription and chain are available in session state
if st.session_state.pdf_file and st.session_state.chain:
    # User input for question
    question = st.text_input("Ask a question about the PDF document:")

    if st.button("Get Answer"):
        # Run the RAG model to get the answer
        answer = st.session_state.chain.invoke(question)
        st.write(f"Answer: {answer}")
