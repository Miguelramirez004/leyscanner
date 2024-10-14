import PyPDF2

# Función para convertir un archivo PDF a texto
def pdf_to_text(pdf_file, txt_file):
    with open(pdf_file, 'rb') as file:
        # Crear un lector de PDF
        reader = PyPDF2.PdfReader(file)
        text = ""
        # Extraer el texto de cada página
        for page in reader.pages:
            text += page.extract_text()
    
    # Guardar el texto extraído en un archivo .txt
    with open(txt_file, 'w', encoding='utf-8') as text_file:
        text_file.write(text)

# Convertir el Código Penal y la Ley de Enjuiciamiento Criminal
pdf_to_text('/mnt/data/BOE-A-1995-25444-consolidado.pdf', 'codigo_penal.txt')
pdf_to_text('/mnt/data/BOE-A-2000-323-consolidado.pdf', 'ley_enjuiciamiento_criminal.txt')

print("Conversión completa: Archivos PDF convertidos a formato .txt")
