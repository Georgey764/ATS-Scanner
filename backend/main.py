from processing_classes import ResumeProcessor

def main():
    file_path = "/Users/george/Documents/Resume/2025/ULM CRM Database Admin/Copy-Resume.docx"
    processor = ResumeProcessor(file_path)

    print(processor.get_similarity_score("""We are looking for a Data Scientist with experience in Python, Machine Learning, 
and Deep Learning. Must be familiar with PyTorch and Scikit-Learn. 
Experience with NLP is a plus."""))

if __name__ == "__main__":
    main()