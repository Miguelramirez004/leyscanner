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
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "codigo_penal_docs" not in st.session_state:
    st.session_state.codigo_penal_docs = []
if "ley_enjuiciamiento_docs" not in st.session_state:
    st.session_state.ley_enjuiciamiento_docs = []
if "context" not in st.session_state:
    st.session_state.context = None

# Entrada para la clave de OpenAI
st.session_state.api_key = st.text_input("Introduce tu OpenAI API key:", st.session_state.api_key, type="password")

# Cargar el PDF de la sentencia judicial
uploaded_file = st.file_uploader("Sube una sentencia en formato PDF:", type="pdf")
if uploaded_file is not None:
    st.session_state.pdf_file = uploaded_file

# Función para cargar las bases de datos legales (Código Penal y LECrim)
def load_legal_database(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        legal_text = file.read()
    return legal_text

# Función para crear documentos a partir de textos
def create_documents_from_text(text, source_name):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata={"source": source_name}) for chunk in chunks]
    return documents

# Cargar los textos legales convertidos (Código Penal y Ley de Enjuiciamiento Criminal)
if not os.path.exists("codigo_penal.txt"):
    st.error("Por favor, asegúrate de tener el archivo 'codigo_penal.txt' en el directorio.")
else:
    codigo_penal = load_legal_database("codigo_penal.txt")
    st.session_state.codigo_penal = codigo_penal

if not os.path.exists("ley_enjuiciamiento_criminal.txt"):
    st.error("Por favor, asegúrate de tener el archivo 'ley_enjuiciamiento_criminal.txt' en el directorio.")
else:
    ley_enjuiciamiento_criminal = load_legal_database("ley_enjuiciamiento_criminal.txt")
    st.session_state.ley_enjuiciamiento_criminal = ley_enjuiciamiento_criminal

# Botón para iniciar el proceso
if st.button("Iniciar Análisis"):
    if not st.session_state.pdf_file or not st.session_state.api_key:
        st.error("Por favor, proporciona un archivo PDF de sentencia válido y tu OpenAI API key.")
    else:
        # Configurar la clave de API de OpenAI
        os.environ['OPENAI_API_KEY'] = st.session_state.api_key

        # Inicializar el modelo de OpenAI
        model = ChatOpenAI(openai_api_key=st.session_state.api_key, model="gpt-3.5-turbo")
        parser = StrOutputParser()

        # Plantilla de prompt para preguntas y respuestas
        qa_template = """
        Usando el contexto proporcionado de las sentencias y leyes aplicables, responde la pregunta de la mejor manera posible.

        Contexto de Sentencias: {sentencias}
        Contexto de Leyes: {leyes}

        Pregunta: {question}
        """
        qa_prompt = ChatPromptTemplate.from_template(qa_template)

        # Extraer el texto del archivo PDF (sentencia)
        pdf_reader = PyPDF2.PdfReader(st.session_state.pdf_file)
        extracted_text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if isinstance(page_text, str):
                extracted_text += page_text
                
        st.session_state.extracted_text = extracted_text
        st.text_area("Texto Extraído de la Sentencia:", st.session_state.extracted_text, height=300)

        # Crear los embeddings a partir de la sentencia
        embeddings = OpenAIEmbeddings()
        sentencia_documents = create_documents_from_text(st.session_state.extracted_text, source_name="sentencia")
        st.session_state.vectorstore_sentencias = DocArrayInMemorySearch.from_documents(sentencia_documents, embeddings)

        if not st.session_state.vectorstore_leyes:
            st.session_state.codigo_penal_docs = create_documents_from_text(st.session_state.codigo_penal, source_name="Código Penal")
            st.session_state.ley_enjuiciamiento_docs = create_documents_from_text(st.session_state.ley_enjuiciamiento_criminal, source_name="Ley de Enjuiciamiento Criminal")
            combined_documents = st.session_state.codigo_penal_docs + st.session_state.ley_enjuiciamiento_docs
            st.session_state.vectorstore_leyes = DocArrayInMemorySearch.from_documents(combined_documents, embeddings)

        # Crear un diccionario de contexto para la cadena RAG
        st.session_state.context = {
            "sentencias": st.session_state.vectorstore_sentencias.as_retriever(),
            "leyes": st.session_state.vectorstore_leyes.as_retriever()
        }

        # Crear la cadena RAG que integra tanto la sentencia como el texto legal
        st.session_state.chain = (
            qa_prompt
            | model
            | parser
        )

# Verificar si el contexto, el texto extraído y la cadena están disponibles en el estado de la sesión
if st.session_state.context and st.session_state.extracted_text and st.session_state.chain:
    # Entrada de usuario para hacer preguntas
    question = st.text_input("Haz una pregunta sobre la sentencia o su relación con el Código Penal / LECrim:")

    if st.button("Obtener Respuesta"):
        try:
            # Crear el input con `context` y `question`
            # Utilizar `get_relevant_documents()` para recuperar documentos de cada vectorstore
            sentencias_relevant = st.session_state.context["sentencias"].get_relevant_documents(question)
            leyes_relevant = st.session_state.context["leyes"].get_relevant_documents(question)

            # Mostrar el contexto recuperado para diagnóstico
            st.write("Contexto Recuperado para Sentencias:")
            for idx, doc in enumerate(sentencias_relevant):
                st.write(f"Fragmento Sentencia {idx + 1}: {doc.page_content}")

            st.write("Contexto Recuperado para Leyes:")
            for idx, doc in enumerate(leyes_relevant):
                st.write(f"Fragmento Ley {idx + 1}: {doc.page_content}")

            # Combinar el contexto de sentencias y leyes en un solo string
            combined_sentencias = "\n".join([doc.page_content for doc in sentencias_relevant])
            combined_leyes = "\n".join([doc.page_content for doc in leyes_relevant])

            # Pasar los contextos individuales como variables separadas al prompt
            answer = st.session_state.chain.invoke({
                "sentencias": combined_sentencias,  # Pasar contexto de sentencias como variable
                "leyes": combined_leyes,            # Pasar contexto de leyes como variable
                "question": question                # Pasar la pregunta como variable
            })
            st.write(f"Respuesta: {answer}")
        except TypeError as e:
            st.error(f"Error al obtener la respuesta: {e}")
            print(f"Error detallado: {e}")
        except Exception as e:
            st.error(f"Se produjo un error inesperado: {e}")
            print(f"Error detallado: {e}")
