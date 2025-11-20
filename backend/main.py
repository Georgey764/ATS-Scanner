from processing_classes import ResumeProcessor
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from typing import Annotated
import csv
from jobspy import scrape_jobs
import re

app = FastAPI()

model = SentenceTransformer('all-MiniLM-L6-v2')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)

def give_scraped_jobs(job_title):
    jobs = scrape_jobs(
        site_name=["indeed", "linkedin", "zip_recruiter", "google"], # "glassdoor", "bayt", "naukri", "bdjobs"
        search_term=job_title,
        google_search_term="software engineer jobs near San Francisco, CA since yesterday",
        location="San Francisco, CA",
        results_wanted=10,
        hours_old=72,
        country_indeed='USA',
        
        # linkedin_fetch_description=True # gets more info such as description, direct job url (slower)
        # proxies=["208.195.175.46:65095", "208.195.175.45:65095", "localhost"],
    )
    # print(f"Found {len(jobs)} jobs")
    # print(jobs.head())
    jobs.to_csv("jobs.csv", quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False)
    return jobs

def clean_text(text):
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

@app.post("/submit-job-application/")
def submit_application(
    resume: Annotated[UploadFile, File()], 
    job_title: Annotated[str, Form()]     
):
    save_path = f"uploads/{resume.filename}"

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(resume.file, buffer)
        
    resume_processor = ResumeProcessor(save_path, model)

    jobs = give_scraped_jobs(job_title)
    best_match_description = clean_text(jobs['description'][0])

    similarity_score = resume_processor.get_similarity_score(best_match_description)

    return {
        "status": "success", 
        "filename": resume.filename, 
        "similarity_score": similarity_score
    }



def main():
    file_path = "/Users/george/Documents/Resume/2025/ULM CRM Database Admin/Copy-Resume.docx"
    processor = ResumeProcessor(file_path)

    print(processor.get_similarity_score("""We are looking for a Data Scientist with experience in Python, Machine Learning, 
    and Deep Learning. Must be familiar with PyTorch and Scikit-Learn. 
    Experience with NLP is a plus."""))

# if __name__ == "__main__":
#     main()
    