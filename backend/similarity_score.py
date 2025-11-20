from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Load the Pre-trained Model
# 'all-MiniLM-L6-v2' is a fast, efficient model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_similarity_score(resume_text, job_description):
    # 2. Encode the text into embeddings (vectors)
    embeddings = model.encode([resume_text, job_description])
    
    resume_vector = embeddings[0]
    jd_vector = embeddings[1]
    
    # 3. Reshape vectors for scikit-learn (needs 2D array)
    resume_vector = resume_vector.reshape(1, -1)
    jd_vector = jd_vector.reshape(1, -1)
    
    # 4. Calculate Cosine Similarity
    score = cosine_similarity(resume_vector, jd_vector)[0][0]
    
    # Convert to percentage
    return round(score * 100, 2)

# --- Example Usage ---

job_desc = """
We are looking for a Data Scientist with experience in Python, Machine Learning, 
and Deep Learning. Must be familiar with PyTorch and Scikit-Learn. 
Experience with NLP is a plus.
"""

resume_A = """
Experienced Software Engineer with a focus on Web Development. 
Proficient in JavaScript, React, and Node.js. 
Basic knowledge of Python scripting.
"""

resume_B = """
Data Scientist with 3 years of experience building ML models. 
Expert in Python, Pandas, and PyTorch. 
Worked on several Natural Language Processing projects.
"""

score_A = get_similarity_score(resume_A, job_desc)
score_B = get_similarity_score(resume_B, job_desc)

print(f"Resume A Match: {score_A}%")
print(f"Resume B Match: {score_B}%")