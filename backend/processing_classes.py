from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import PyPDF2
import docx2txt
import re

class ResumeProcessor:
    def _extract_text_from_file(self, file_path):
        print("Extracting text from file...")
        text = ""
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text()
        elif file_path.endswith('.docx'):
            text = docx2txt.process(file_path)
        return text

    def _clean_text(self, text):
        print("Cleaning up extracted text...")
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
    
    def __init__(self, file_name):
        print("Initializing AI model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        text = self._extract_text_from_file(file_name)
        self.resume_text = self._clean_text(text)
        
    def get_similarity_score(self, job_description):
        print("Calculating Similarity Score")
        embeddings = self.model.encode([self.resume_text, job_description])
        
        resume_vector = embeddings[0]
        jd_vector = embeddings[1]

        resume_vector = resume_vector.reshape(1, -1)
        jd_vector = jd_vector.reshape(1, -1)
        
        score = cosine_similarity(resume_vector, jd_vector)[0][0]
        
        return round(score * 100, 2)
        
    