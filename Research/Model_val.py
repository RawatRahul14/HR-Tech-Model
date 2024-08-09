import streamlit as st
import fitz  # PyMuPDF for PDF processing
import spacy
from spacy.matcher import PhraseMatcher
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the skills list
def load_skills_from_file(file_path='unique_skills.txt'):
    with open(file_path, 'r') as file:
        skills = [line.strip() for line in file]
    return skills

# Function to extract text from a PDF resume
def extract_text_from_pdf(pdf_bytes):
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error opening or reading PDF: {e}")
        return None

# Function to extract skills and key points from text
def extract_skills_and_points(text, skill_phrases):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    skills = []
    key_points = []

    matcher = PhraseMatcher(nlp.vocab)
    patterns = [nlp(skill) for skill in skill_phrases]
    matcher.add("SKILLS", patterns)

    # Use matcher to find skills
    matches = matcher(doc)
    for match_id, start, end in matches:
        skills.append(doc[start:end].text)

    # Extract key points
    for token in doc:
        if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
            key_points.append(token.text)

    return skills, key_points

# Load the vectorizer and job data
def load_vectorizer_and_data(vectorizer_path='Data/vectorizer.pkl', data_path='Data/job_data.pkl'):
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(data_path, 'rb') as f:
        df = pickle.load(f)
    return vectorizer, df

# Function to recommend top 3 jobs
def recommend_top_3(job_tags, gender, qualification, experience, vectorizer, df):
    # Combine job tags with gender, qualification, and experience
    combined_input = f"{job_tags}, {gender}, {qualification}, {experience}"

    # Step 1: Vectorize the 'tags' column using the loaded vectorizer
    tfidf_matrix = vectorizer.transform(df['tags'])
    
    # Step 2: Vectorize the combined input
    input_vector = vectorizer.transform([combined_input])
    
    # Step 3: Compute cosine similarity
    cosine_similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    
    # Step 4: Get indices of the top similarity scores, allowing for potential duplicates
    top_indices = cosine_similarities.argsort()[-len(df):][::-1]
    
    # Step 5: Extract job titles and ensure they are unique
    top_jobs = []
    seen_jobs = set()
    
    for idx in top_indices:
        job_title = df.iloc[idx]['Job Title']
        if job_title not in seen_jobs:
            top_jobs.append(job_title)
            seen_jobs.add(job_title)
        if len(top_jobs) == 3:
            break
    
    # Ensure we return 3 jobs, if available
    if len(top_jobs) < 3:
        top_jobs += ['No more distinct jobs available'] * (3 - len(top_jobs))
    
    return top_jobs

def main():
    st.title("Job Recommendation System")

    # Load vectorizer and job data
    vectorizer, df = load_vectorizer_and_data()
    skill_phrases = load_skills_from_file()

    st.header("Upload your Resume")
    resume_pdf = st.file_uploader("Upload Resume (PDF)", type="pdf")
    
    if resume_pdf is not None:
        resume_bytes = resume_pdf.read()
        resume_text = extract_text_from_pdf(resume_bytes)
        if resume_text:
            st.write("Resume Text:")
            st.write(resume_text)

            # Extract skills and key points from resume
            skills, key_points = extract_skills_and_points(resume_text, skill_phrases)
            st.write("Skills extracted from Resume:")
            st.write(list(set(skills)))
            
            # Input for gender
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])

            # Input for highest degree
            highest_degree = st.selectbox("Highest Degree", ["High School", "Associate Degree", "Bachelor's Degree", "Master's Degree", "Doctorate", "Other"])

            # Input for skills and experience
            num_skills = st.number_input("Number of Skills", min_value=1, step=1, value=1)
            skills_experience = {}
            total_experience = 0
            for i in range(num_skills):
                skill = st.text_input(f"Skill {i+1}")
                experience = st.number_input(f"Experience in years for {skill}", min_value=0, step=1, value=0)
                skills_experience[skill] = experience
                total_experience += experience
            
            # Use extracted skills to recommend jobs
            job_tags = ', '.join(skills)
            recommended_jobs = recommend_top_3(job_tags, gender, highest_degree, total_experience, vectorizer, df)
            
            st.write("Top 3 similar jobs based on extracted skills, qualification, and experience:")
            for job in recommended_jobs:
                st.write(f"- {job}")

if __name__ == "__main__":
    main()
