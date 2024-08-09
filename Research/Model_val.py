import streamlit as st
from io import BytesIO
import fitz  # PyMuPDF
import spacy
from spacy.matcher import PhraseMatcher

# Skill extraction functions
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

def extract_skills_and_points(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    skills = []
    key_points = []

    # Define some common skills to search for
    skill_phrases = [
        "machine learning", "deep learning", "data analysis", "python", "java",
        "project management", "communication", "teamwork", "sql", "excel"
    ]
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

def main():
    st.title("Job Recommendation System")
    
    st.header("Upload your Resume")
    resume_pdf = st.file_uploader("Upload Resume (PDF)", type="pdf")
    
    if resume_pdf is not None:
        # Extract text and skills from the resume
        resume_bytes = resume_pdf.read()
        resume_text = extract_text_from_pdf(resume_bytes)
        if resume_text:
            skills, key_points = extract_skills_and_points(resume_text)
            st.write("Skills extracted from Resume:")
            st.write(skills)
    
    st.header("Personal Information")
    
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    st.header("Skills and Experience")
    num_skills = st.number_input("Number of Skills", min_value=1, step=1, value=1)
    
    skills_experience = {}
    for i in range(num_skills):
        skill = st.text_input(f"Skill {i+1}")
        experience = st.number_input(f"Experience in years for {skill}", min_value=0, step=1, value=0)
        skills_experience[skill] = experience
    
    if st.button("Submit"):
        st.write("Submitted Information:")
        st.write("Resume: Uploaded" if resume_pdf else "Resume: Not Uploaded")
        st.write(f"Gender: {gender}")
        st.write("Skills and Experience:")
        st.write(skills_experience)
        
        # You can then pass this information to your job title prediction model
        # job_title_prediction = predict_job_title(resume_bytes, gender, skills_experience)
        # st.write(f"Predicted Job Title: {job_title_prediction}")

if __name__ == "__main__":
    main()
