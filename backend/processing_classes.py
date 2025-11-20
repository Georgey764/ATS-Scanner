from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import PyPDF2
import docx2txt
import re
import csv
from jobspy import scrape_jobs

def give_scraped_jobs(job_title):
    jobs = scrape_jobs(
        site_name=["indeed", "linkedin", "zip_recruiter", "google"], # "glassdoor", "bayt", "naukri", "bdjobs"
        search_term=job_title,
        google_search_term="software engineer jobs near San Francisco, CA since yesterday",
        location="San Francisco, CA",
        results_wanted=20,
        hours_old=72,
        country_indeed='USA',
        
        # linkedin_fetch_description=True # gets more info such as description, direct job url (slower)
        # proxies=["208.195.175.46:65095", "208.195.175.45:65095", "localhost"],
    )
    print(f"Found {len(jobs)} jobs")
    print(jobs.head())
    jobs.to_csv("jobs.csv", quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False)

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
    
    def __init__(self, file_name, model):
        self.model = model
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
        
    