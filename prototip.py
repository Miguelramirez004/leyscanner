import os
import streamlit as st
import PyPDF2
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema import Document

# Configurar el entorno de codificación
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Función para inicializar la UI
def initialize_ui():
    st.sidebar.title("Análisis Legal con LLM")
    st.sidebar.markdown(
        """
        Esta aplicación permite analizar sentencias judiciales en español
        utilizando modelos de lenguaje. Proporcione una sentencia en PDF y la API key de OpenAI para comenzar.
        """
    )
    # Inputs en la sidebar
    st.session_state.api_key = st.sidebar.text_input("Introduce tu OpenAI API key:", type="password", value=st.session_state.get("api_key", ""))
    st.session_state.pdf_file = st.sidebar.file_uploader("Sube una sentencia en formato PDF:", type="pdf")
    iniciar_analisis = st.sidebar.button("Iniciar Análisis")
    return iniciar_analisis

# Función para cargar la base de datos legal (Código Penal y LECrim)
def load_legal_database(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Función para extraer texto de un PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    return "".join([page.extract_text() for page in pdf_reader.pages if isinstance(page.extract_text(), str)])

# Función para crear documentos a partir del texto
def create_documents(text, source_name):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"source": source_name}) for chunk in chunks]

# Función para mostrar la respuesta en un diseño más organizado
def display_response(context_sentencias, context_leyes, answer):
    with st.expander("Contexto Recuperado para Sentencias", expanded=False):
        for idx, doc in enumerate(context_sentencias):
            st.markdown(f"**Fragmento Sentencia {idx + 1}:** {doc.page_content}")

    with st.expander("Contexto Recuperado para Leyes", expanded=False):
        for idx, doc in enumerate(context_leyes):
            st.markdown(f"**Fragmento Ley {idx + 1}:** {doc.page_content}")

    st.subheader("Respuesta Generada")
    st.markdown(f"**{answer}**")

# Inicializar la UI y obtener entradas del usuario
iniciar_analisis = initialize_ui()

# Cargar los archivos legales requeridos
if "codigo_penal" not in st.session_state:
    st.session_state.codigo_penal = load_legal_database("codigo_penal.txt") if os.path.exists("codigo_penal.txt") else None
if "ley_enjuiciamiento_criminal" not in st.session_state:
    st.session_state.ley_enjuiciamiento_criminal = load_legal_database("ley_enjuiciamiento_criminal.txt") if os.path.exists("ley_enjuiciamiento_criminal.txt") else None

# Mostrar errores si faltan archivos legales
if not st.session_state.codigo_penal or not st.session_state.ley_enjuiciamiento_criminal:
    st.sidebar.error("Por favor, asegúrate de tener los archivos 'codigo_penal.txt' y 'ley_enjuiciamiento_criminal.txt' en el directorio.")

# Al hacer clic en "Iniciar Análisis", procesar el archivo PDF
if iniciar_analisis and st.session_state.pdf_file and st.session_state.api_key:
    st.session_state.api_key = st.session_state.api_key  # Almacenar la API key en session_state

    # Configurar la clave de API de OpenAI
    os.environ['OPENAI_API_KEY'] = st.session_state.api_key

    # Inicializar el modelo de OpenAI
    model = ChatOpenAI(openai_api_key=st.session_state.api_key, model="gpt-3.5-turbo")
    parser = StrOutputParser()
    embeddings = OpenAIEmbeddings()

    # Extraer el texto del archivo PDF y crear documentos
    st.session_state.extracted_text = extract_text_from_pdf(st.session_state.pdf_file)
    sentencia_documents = create_documents(st.session_state.extracted_text, source_name="sentencia")

    # Crear vectorstore para la sentencia y las leyes
    st.session_state.vectorstore_sentencias = DocArrayInMemorySearch.from_documents(sentencia_documents, embeddings)
    codigo_penal_docs = create_documents(st.session_state.codigo_penal, source_name="Código Penal")
    ley_enjuiciamiento_docs = create_documents(st.session_state.ley_enjuiciamiento_criminal, source_name="Ley de Enjuiciamiento Criminal")
    st.session_state.vectorstore_leyes = DocArrayInMemorySearch.from_documents(codigo_penal_docs + ley_enjuiciamiento_docs, embeddings)

    # Crear el contexto de la aplicación
    st.session_state.context = {"sentencias": st.session_state.vectorstore_sentencias.as_retriever(), "leyes": st.session_state.vectorstore_leyes.as_retriever()}

    # Crear la cadena RAG con el prompt
    qa_template = """
    Usando el contexto proporcionado de las sentencias y leyes aplicables, responde la pregunta de la mejor manera posible.

    Contexto de Sentencias: {sentencias}
    Contexto de Leyes: {leyes}

    Pregunta: {question}
    """
    st.session_state.qa_prompt = ChatPromptTemplate.from_template(qa_template)
    st.session_state.chain = (st.session_state.qa_prompt | model | parser)

    # Mensaje informativo
    st.success("Análisis iniciado correctamente. Ahora puedes realizar preguntas sobre la sentencia.")

# Mostrar el área de texto para la pregunta y el botón de obtener respuesta si el análisis está listo
if "extracted_text" in st.session_state:
    st.text_area("Texto Extraído de la Sentencia:", st.session_state.extracted_text, height=200)
    question = st.text_input("Haz una pregunta sobre la sentencia o su relación con el Código Penal / LECrim:")

    if st.button("Obtener Respuesta") and st.session_state.context and question:
        # Recuperar documentos relevantes del contexto
        sentencias_relevant = st.session_state.context["sentencias"].get_relevant_documents(question)
        leyes_relevant = st.session_state.context["leyes"].get_relevant_documents(question)

        # Combinar contexto de sentencias y leyes
        combined_sentencias = "\n".join([doc.page_content for doc in sentencias_relevant])
        combined_leyes = "\n".join([doc.page_content for doc in leyes_relevant])

        # Generar la respuesta con el contexto
        answer = st.session_state.chain.invoke({
            "sentencias": combined_sentencias,
            "leyes": combined_leyes,
            "question": question
        })

        # Mostrar la respuesta en la interfaz
        display_response(sentencias_relevant, leyes_relevant, answer)
