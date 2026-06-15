import fitz  # PyMuPDF
import zipfile
import xml.etree.ElementTree as ET

def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a PDF file.
    """
    text = ""
    try:
        # Open the PDF document
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text

def extract_text_from_docx(docx_path):
    """
    Extracts all text from a DOCX file using pure Python XML parsing.
    """
    text = ""
    try:
        with zipfile.ZipFile(docx_path) as docx:
            xml_content = docx.read('word/document.xml')
            root = ET.fromstring(xml_content)
            # Find all text elements
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            paragraphs = root.findall('.//w:t', ns)
            text = "\n".join([p.text for p in paragraphs if p.text])
    except Exception as e:
        print(f"Error reading DOCX {docx_path}: {e}")
    return text
