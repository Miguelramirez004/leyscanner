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
from langchain.schema import Document  # Importar la clase Document

# Streamlit UI elements
st.title("Análisis Legal de Sentencias Judiciales con Referencias al Código Penal y LECrim")

# Inicializar variables de sesión para las sentencias y leyes
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "pdf_file" not in st.session_state:
    st.session_state.pdf_file = None
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "vectorstore_sentencias" not in st.session_state:
    st.session_state.vectorstore_sentencias = None
if "vectorstore_leyes" not in st.session_state:
    st.session_state.vectorstore_leyes = None
if "chain" not in st.session_state:
    st.session_state.chain = None

# Entrada para la clave de OpenAI
st.session_state.api_key = st.text_input("Introduce tu OpenAI API key:", st.session_state.api_key, type="password")

# Asegúrate de que la clave API esté presente antes de proceder
if not st.session_state.api_key:
    st.error("Por favor, introduce una OpenAI API Key válida para continuar.")
    st.stop()
else:
    # Configura la variable de entorno con la clave de API ingresada
    os.environ['OPENAI_API_KEY'] = st.session_state.api_key

# Cargar el PDF de la sentencia judicial
uploaded_file = st.file_uploader("Sube una sentencia en formato PDF:", type="pdf")
if uploaded_file is not None:
    st.session_state.pdf_file = uploaded_file

# Función para cargar las bases de datos legales
def load_legal_database(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        legal_text = file.read()
    return legal_text

# Función para convertir un archivo PDF a texto
def pdf_to_text(pdf_file, txt_file):
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    # Guardar el texto extraído en un archivo .txt
    with open(txt_file, 'w', encoding='utf-8') as text_file:
        text_file.write(text)

# Convertir el PDF del Código Penal y Ley de Enjuiciamiento Criminal si no existen los archivos .txt
if not os.path.exists("codigo_penal.txt"):
    st.warning("Convirtiendo el PDF del Código Penal a texto...")
    pdf_to_text("/mnt/data/BOE-A-1995-25444-consolidado.pdf", "codigo_penal.txt")

if not os.path.exists("ley_enjuiciamiento_criminal.txt"):
    st.warning("Convirtiendo el PDF de la Ley de Enjuiciamiento Criminal a texto...")
    pdf_to_text("/mnt/data/BOE-A-2000-323-consolidado.pdf", "ley_enjuiciamiento_criminal.txt")

# Cargar los textos legales convertidos
codigo_penal = load_legal_database("codigo_penal.txt")
ley_enjuiciamiento_criminal = load_legal_database("ley_enjuiciamiento_criminal.txt")

# Crear vector stores para las leyes usando la clase Document
def create_documents_from_text(text, source_name):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)

    # Crear objetos de tipo Document a partir de cada fragmento de texto
    documents = [Document(page_content=chunk, metadata={"source": source_name}) for chunk in chunks]
    return documents

# Crear embeddings y vector stores si no se han creado
if st.session_state.vectorstore_leyes is None:
    # Inicializar los embeddings aquí, antes de cualquier uso
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.api_key)  # Inicializar embeddings con la API key

    # Crear documentos para el Código Penal y Ley de Enjuiciamiento Criminal
    codigo_penal_documents = create_documents_from_text(codigo_penal, source_name="Código Penal")
    ley_enjuiciamiento_criminal_documents = create_documents_from_text(ley_enjuiciamiento_criminal, source_name="Ley de Enjuiciamiento Criminal")

    # Combinar todos los documentos en una lista
    combined_documents = codigo_penal_documents + ley_enjuiciamiento_criminal_documents

    # Crear un único vector store a partir de todos los documentos combinados
    st.session_state.vectorstore_leyes = DocArrayInMemorySearch.from_documents(combined_documents, embeddings)

# Continuar con el procesamiento del archivo de la sentencia
if st.button("Iniciar Análisis"):
    if not st.session_state.pdf_file:
        st.error("Por favor, proporciona un archivo PDF de sentencia válido.")
    else:
        # Configurar el cliente de OpenAI con la API key
        model = ChatOpenAI(openai_api_key=st.session_state.api_key, model="gpt-3.5-turbo")
        parser = StrOutputParser()

        # Leer el PDF de la sentencia
        pdf_reader = PyPDF2.PdfReader(st.session_state.pdf_file)
        extracted_text = "".join([page.extract_text() for page in pdf_reader.pages])
        st.session_state.extracted_text = extracted_text

        # Crear objetos de tipo Document a partir de la sentencia
        sentencia_documents = [Document(page_content=chunk, metadata={"source": "sentencia"})
                               for chunk in RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20).split_text(st.session_state.extracted_text)]

        # Crear vector store para la sentencia
        sentencia_vectorstore = DocArrayInMemorySearch.from_documents(sentencia_documents, embeddings)
        st.session_state.vectorstore_sentencias = sentencia_vectorstore

        # Crear la cadena de preguntas y respuestas integrando las leyes y la sentencia
        qa_template = """
        Responde la pregunta basada en el contexto proporcionado. Si no puedes 
        responder a la pregunta, contesta "No lo sé".

        Contexto: {context}

        Pregunta: {question}
        """
        qa_prompt = ChatPromptTemplate.from_template(qa_template)

        st.session_state.chain = (
            {"context": [st.session_state.vectorstore_sentencias.as_retriever(), st.session_state.vectorstore_leyes.as_retriever()], "question": RunnablePassthrough()}
            | qa_prompt
            | model
            | parser
        )

# Preguntar sobre la sentencia y las leyes
if st.session_state.extracted_text and st.session_state.chain:
    question = st.text_input("Haz una pregunta sobre la sentencia o su relación con el Código Penal / LECrim:")

    if st.button("Obtener Respuesta"):
        answer = st.session_state.chain.invoke(question)
        st.write(f"Respuesta: {answer}")
