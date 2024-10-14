import os
import streamlit as st
import PyPDF2

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

# Verificar si los archivos PDF existen, si no, subirlos nuevamente
if not os.path.exists("BOE-A-1995-25444-consolidado.pdf") or not os.path.exists("BOE-A-2000-323-consolidado.pdf"):
    st.warning("Los archivos PDF no se encuentran en el directorio. Por favor, vuelve a subirlos.")
    
    # Permitir que el usuario suba los archivos nuevamente
    codigo_penal_pdf = st.file_uploader("Sube el archivo PDF del Código Penal:", type="pdf", key="codigo_penal_pdf")
    ley_enjuiciamiento_pdf = st.file_uploader("Sube el archivo PDF de la Ley de Enjuiciamiento Criminal:", type="pdf", key="ley_enjuiciamiento_pdf")
    
    # Guardar los archivos subidos en el directorio actual
    if codigo_penal_pdf is not None:
        with open("BOE-A-1995-25444-consolidado.pdf", "wb") as f:
            f.write(codigo_penal_pdf.getbuffer())
        st.success("Archivo del Código Penal cargado correctamente.")

    if ley_enjuiciamiento_pdf is not None:
        with open("BOE-A-2000-323-consolidado.pdf", "wb") as f:
            f.write(ley_enjuiciamiento_pdf.getbuffer())
        st.success("Archivo de la Ley de Enjuiciamiento Criminal cargado correctamente.")

# Convertir PDF a TXT (para el Código Penal y LECrim)
def convert_pdf_to_txt(pdf_path, txt_path):
    with open(pdf_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(text)

# Convertir el PDF del Código Penal y Ley de Enjuiciamiento Criminal si no existen los archivos .txt
if os.path.exists("BOE-A-1995-25444-consolidado.pdf") and not os.path.exists("codigo_penal.txt"):
    convert_pdf_to_txt("BOE-A-1995-25444-consolidado.pdf", "codigo_penal.txt")
    st.success("Archivo 'codigo_penal.txt' creado exitosamente.")

if os.path.exists("BOE-A-2000-323-consolidado.pdf") and not os.path.exists("ley_enjuiciamiento_criminal.txt"):
    convert_pdf_to_txt("BOE-A-2000-323-consolidado.pdf", "ley_enjuiciamiento_criminal.txt")
    st.success("Archivo 'ley_enjuiciamiento_criminal.txt' creado exitosamente.")

# Comprobar que los archivos TXT se crearon correctamente
if os.path.exists("codigo_penal.txt") and os.path.exists("ley_enjuiciamiento_criminal.txt"):
    st.success("Los archivos de texto para el Código Penal y la Ley de Enjuiciamiento Criminal están listos para usarse.")
else:
    st.error("Por favor, asegúrate de que los archivos PDF se han convertido correctamente a .txt.")
