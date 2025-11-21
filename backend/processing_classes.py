from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import PyPDF2
import docx2txt
import re

class ResumeProcessor:
    def _normalize_score(self, raw_score):
        min_threshold = 0.20
        max_threshold = 0.7

        if raw_score < min_threshold:
            return 0.0
        if raw_score > max_threshold:
            return 100.0
        
        normalized = (raw_score - min_threshold) / (max_threshold - min_threshold)
        
        return round(normalized * 100, 2)
    
    def _chunk_text(self, text, chunk_size=250, overlap=50):
        words = text.split()
        chunks = []
        
        # If resume is short, just return it as one chunk
        if len(words) <= chunk_size:
            return [text]
        
        # Create sliding window chunks
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks
    
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

        text = text.lower()

        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'http\S+', '', text)
        
        text = re.sub(r'[^a-zA-Z0-9\s\+\#\.\-]', '', text)

        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def __init__(self, file_name, model):
        self.model = model
        text = self._extract_text_from_file(file_name)
        self.resume_text = self._clean_text(text)
        
    def get_similarity_score(self, job_description):
        try:
            print("Calculating Symmetric Similarity...")
            
            resume_chunks = self._chunk_text(self.resume_text, chunk_size=250, overlap=50)
            
            jd_chunks = self._chunk_text(job_description, chunk_size=250, overlap=50)
            
            print(f"Resume Chunks: {len(resume_chunks)} | JD Chunks: {len(jd_chunks)}")

            resume_vectors = self.model.encode(resume_chunks)
            jd_vectors = self.model.encode(jd_chunks)

            similarity_matrix = cosine_similarity(resume_vectors, jd_vectors)

            best_matches_for_jd = np.max(similarity_matrix, axis=0)
            
            model_score = round(np.mean(best_matches_for_jd), 2)
            final_score = self._normalize_score(model_score)
            
            print(f"Symmetric Score: {model_score}")
            return final_score

        except Exception as e:
            print(f"Error: {e}")
            raise e

        
    
   
        
    