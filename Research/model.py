import streamlit as st
from io import BytesIO
import fitz  # PyMuPDF
import spacy
from spacy.matcher import PhraseMatcher
import pickle  # For loading the pickle model
import numpy as np

# Load the skills list
def load_skills_from_file(file_path='unique_skills.txt'):
    with open(file_path, 'r') as file:
        skills = [line.strip() for line in file]
    return skills

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

# Function to preprocess inputs for the model
def preprocess_inputs(gender, highest_degree, skills_experience, skill_phrases):
    # One-hot encode gender
    gender_dict = {"Male": [1, 0, 0], "Female": [0, 1, 0], "Other": [0, 0, 1]}
    gender_encoded = gender_dict[gender]

    # Encode highest degree
    degree_dict = {
        "High School": 0, "Associate Degree": 1, "Bachelor's Degree": 2,
        "Master's Degree": 3, "Doctorate": 4, "Other": 5
    }
    degree_encoded = degree_dict[highest_degree]

    # Create skills experience vector
    skills_vector = np.zeros(len(skill_phrases))
    for i, skill in enumerate(skill_phrases):
        if skill in skills_experience:
            skills_vector[i] = skills_experience[skill]

    # Combine all inputs into a single input vector
    input_vector = np.concatenate((gender_encoded, [degree_encoded], skills_vector))

    return input_vector.reshape(1, -1)  # Reshape for model input

# Function to predict job title using the pickle model
def predict_job_title(model, input_vector):
    prediction = model.predict(input_vector)
    predicted_job_title = np.argmax(prediction, axis=1)  # Assuming it's a classification model
    return predicted_job_title

def main():
    st.title("Job Recommendation System")

    # Load the pickle model
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Load skills from file or list
    skill_phrases = load_skills_from_file()  # This can be modified if you load skills from a variable directly
    
    st.header("Upload your Resume")
    resume_pdf = st.file_uploader("Upload Resume (PDF)", type="pdf")
    
    if resume_pdf is not None:
        # Extract text and skills from the resume
        resume_bytes = resume_pdf.read()
        resume_text = extract_text_from_pdf(resume_bytes)
        if resume_text:
            skills, key_points = extract_skills_and_points(resume_text, skill_phrases)
            st.write("Skills extracted from Resume:")
            st.write(skills)
    
    st.header("Personal Information")
    
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    st.header("Education")
    highest_degree = st.selectbox("Highest Degree", ["High School", "Associate Degree", "Bachelor's Degree", "Master's Degree", "Doctorate", "Other"])
    
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
        st.write(f"Highest Degree: {highest_degree}")
        st.write("Skills and Experience:")
        st.write(skills_experience)

        # Preprocess inputs and predict job title
        input_vector = preprocess_inputs(gender, highest_degree, skills_experience, skill_phrases)
        predicted_job_title = predict_job_title(model, input_vector)
        st.write(f"Predicted Job Title: {predicted_job_title}")

if __name__ == "__main__":
    main()
