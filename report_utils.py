
import io
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import re

def clean_markdown(text):
    """
    Simples markdown cleaner for Word.
    Removes bold/italic markers but keeps the text.
    """
    if not text: return ""
    # Remove bold/italic ** or *
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    return text

def generate_word_report(results):
    """
    Generates a Word document from the analysis results.
    results: dict { "Section Name": "Content Text", ... }
    """
    doc = Document()
    
    # Title
    title = doc.add_heading('수주비책 (Win Strategy) 분석 보고서', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Set Korean Font Style
    from docx.oxml.ns import qn
    
    style = doc.styles['Normal']
    style.font.name = 'Malgun Gothic'
    style._element.rPr.rFonts.set(qn('w:eastAsia'), 'Malgun Gothic')
    
    for heading in ['Heading 1', 'Heading 2', 'Heading 3']:
        style = doc.styles[heading]
        style.font.name = 'Malgun Gothic'
        style._element.rPr.rFonts.set(qn('w:eastAsia'), 'Malgun Gothic')
    
    for section, content in results.items():
        if not content: continue
        
        # Section Header
        doc.add_heading(section, level=1)
        
        # Process Content (Line by line to handle basics)
        # This is a basic implementation. For complex markdown tables, 
        # it will write the raw markdown text.
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('### '):
                doc.add_heading(clean_markdown(line[4:]), level=3)
            elif line.startswith('## '):
                doc.add_heading(clean_markdown(line[3:]), level=2)
            elif line.startswith('# '):
                doc.add_heading(clean_markdown(line[2:]), level=1)
            elif line.startswith('- ') or line.startswith('* '):
                p = doc.add_paragraph(clean_markdown(line[2:]), style='List Bullet')
            elif line.startswith('1. '):
                # Simple check for numbered list
                p = doc.add_paragraph(clean_markdown(line[3:]), style='List Number')
            else:
                doc.add_paragraph(clean_markdown(line))
                
    # Save to IO buffer
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer
