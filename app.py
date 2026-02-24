from flask import Flask, render_template, request
import PyPDF2
import re

# ML Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# -------------------------
# Extract text from PDF
# -------------------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# -------------------------
# AI MATCHING USING TF-IDF
# -------------------------
def calculate_match(resume_text, job_description):
    documents = [resume_text, job_description]

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    match_percentage = round(float(similarity[0][0]) * 100, 2)

    return match_percentage


# -------------------------
# AI ROLE SUGGESTION
# -------------------------
def suggest_roles(resume_text):

    roles = {
        "Python Developer":
        "Develops backend applications using Python, Flask, Django, APIs, and SQL databases.",
        
        "Data Analyst":
        "Analyzes data using Excel, SQL, Python, pandas, visualization and reporting tools.",
        
        "Web Developer":
        "Builds websites using HTML, CSS, JavaScript, frontend frameworks and backend integration.",
        
        "IT Support Specialist":
        "Provides technical support, troubleshooting, networking, hardware and Windows system support.",
        
        "Cyber Security Analyst":
        "Works on network security, firewalls, vulnerability assessment, encryption and cyber protection.",
        
        "Business Analyst":
        "Analyzes business processes, requirements gathering, documentation, stakeholder communication and strategy.",
        
        "Project Manager":
        "Manages projects, planning, budgeting, leadership, team coordination and risk management.",
        
        "Digital Marketing Executive":
        "Handles SEO, social media marketing, online campaigns, analytics and digital branding."
    }

    role_scores = {}

    for role, description in roles.items():
        documents = [resume_text, description]

        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)

        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

        score = round(float(similarity[0][0]) * 100, 2)

        role_scores[role] = score

    # Sort roles by highest score
    sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_roles


# -------------------------
# MAIN ROUTE
# -------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    match = None
    roles = None
    missing = []

    if request.method == 'POST':
        resume_file = request.files['resume']
        job_description = request.form['job_description']

        resume_text = extract_text_from_pdf(resume_file)

        match = calculate_match(resume_text, job_description)
        roles = suggest_roles(resume_text)

    return render_template('index.html', match=match, missing=missing, roles=roles)


if __name__ == '__main__':
    app.run(debug=True)