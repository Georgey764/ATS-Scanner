import PyPDF2
import docx2txt
import re

def extract_text_from_file(file_path):
    text = ""
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
    elif file_path.endswith('.docx'):
        text = docx2txt.process(file_path)
    return text

def clean_text(text):
    # 1. Lowercase everything
    text = text.lower()
    
    # 2. Remove emails and URLs (they add noise)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    
    # 3. Remove special characters and numbers (keep text only)
    # This helps focus on the semantic meaning of words
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 4. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

file_path = "/Users/george/Documents/Resume/2025/ULM CRM Database Admin/Copy-Resume.docx"

text = extract_text_from_file(file_path)
clean_text = clean_text(text)

print("text=" + clean_text)