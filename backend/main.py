from processing_classes import ResumeProcessor
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from typing import Annotated
import csv
from jobspy import scrape_jobs
import re
import pandas as pd

app = FastAPI()

model = SentenceTransformer('all-mpnet-base-v2')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)

def delete_file(file_to_delete):
    try:
        if os.path.exists(file_to_delete):
            os.remove(file_to_delete)
            print(f"File '{file_to_delete}' removed successfully.")
        else:
            print(f"File '{file_to_delete}' not found.")

    except PermissionError:
        print(f"Error: Permission denied. Cannot remove '{file_to_delete}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def give_scraped_jobs(job_title, city="San Fransisco", state="California"):
    jobs = scrape_jobs(
        site_name=["indeed", "linkedin"], # "google", "zip_recruiter", "glassdoor", "bayt", "naukri", "bdjobs"
        search_term=job_title,
        # google_search_term=f"{job_title} jobs near San Francisco, CA since yesterday",
        location=city + ", " + state,
        results_wanted=5,
        hours_old=96,
        country_indeed="USA",
        linkedin_fetch_description=True
        
        # linkedin_fetch_description=True # gets more info such as description, direct job url (slower)
        # proxies=["208.195.175.46:65095", "208.195.175.45:65095", "localhost"],
    )
    # print(f"Found {len(jobs)} jobs")
    # print(jobs.head())
    jobs.to_csv("jobs.csv", quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False)
    return jobs

def clean_text(text):
    print("Cleaning up extracted text...")
    
    # 1. Lowercase (Optional, but safe)
    text = text.lower()
    
    # 2. Remove emails and URLs (Good, these are noise)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    
    # We keep: a-z, 0-9, space, +, #, ., and -
    text = re.sub(r'[^a-zA-Z0-9\s\+\#\.\-]', '', text)
    
    # 4. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def nan_to_text(value):
    if pd.isna(value):
        return ""
    
    if isinstance(value, (int, float)):
        return str(value)
        
    return value

@app.post("/submit-job-application/")
def submit_application(
    resume: Annotated[UploadFile, File()], 
    job_title: Annotated[str, Form()],
    city: Annotated[str, Form()],
    state: Annotated[str, Form()]
):
    save_path = f"uploads/{resume.filename}"

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(resume.file, buffer)
        
    try:
        resume_processor = ResumeProcessor(save_path, model)

        jobs = give_scraped_jobs(job_title, city, state)
        result = []
        
        if "description" in jobs.columns:
            for i in range(0, len(jobs["description"])):
                description = nan_to_text(jobs["description"][i])
                print("description:"+ description)
                result_dict = {
                    "similarity_score":resume_processor.get_similarity_score(description),
                    "description": description,
                    "job_link": nan_to_text(jobs["job_url_direct"][i]),
                    "title": nan_to_text(jobs["title"][i]),
                    "company": nan_to_text(jobs["company"][i])
                }
                result.append(result_dict)
        else:
            return {
                "status": "no jobs found in the area"
            }
        
        delete_file(file_to_delete=save_path)
        
        return {
            "status": "success", 
            "filename": resume.filename, 
            "result": result
        }
    except Exception as e:
        print(f"Error in calculation: {e}")
        raise HTTPException(status_code=500, detail=f"Similarity calculation failed: {str(e)}")
    