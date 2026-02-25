from docling.document_converter import DocumentConverter

def parse(source: str):
    converter = DocumentConverter()
    doc = converter.convert(source).document
    return doc.export_to_text()