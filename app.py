from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
import joblib
import numpy as np
import os
import requests
from google import genai
from google.genai import types
from database import init_db, save_consent, save_prediction, get_all_predictions, register_user, login_user

app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)

init_db()

model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

field_mapping = {
    'STEM': 0, 'Health': 1, 'Engineering': 2,
    'Business': 3, 'Arts': 4, 'Education': 5
}

institution_mapping = {
    'Traditional University': 0,
    'University of Technology': 1,
    'TVET College': 2
}

@app.route('/')
def home():
    return jsonify({'message': 'Career Guidance API is running!'})

@app.route('/home')
def home_page():
    return send_from_directory('.', 'platform.html')

@app.route('/platform')
def platform():
    return send_from_directory('.', 'platform.html')

@app.route('/login-page')
def login_page():
    return send_from_directory('.', 'login.html')

@app.route('/register-page')
def register_page():
    return send_from_directory('.', 'register.html')

@app.route('/dashboard')
def dashboard():
    return send_from_directory('.', 'dashboard.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    name = data.get('name', '')
    email = data.get('email', '')
    password = data.get('password', '')
    if not name or not email or not password:
        return jsonify({'success': False, 'message': 'All fields are required'})
    success = register_user(name, email, password)
    if success:
        return jsonify({'success': True, 'message': 'Account created successfully!'})
    else:
        return jsonify({'success': False, 'message': 'Email already exists. Please login.'})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email', '')
    password = data.get('password', '')
    user = login_user(email, password)
    if user:
        return jsonify({'success': True, 'message': 'Login successful!', 'name': user[1]})
    else:
        return jsonify({'success': False, 'message': 'Invalid email or password.'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    name = data.get('name', 'Anonymous')
    nqf_level = data.get('nqf_level', 7)
    duration = data.get('duration', 3)
    skills_count = data.get('skills_count', 3)
    field = data.get('field', 'STEM')
    institution = data.get('institution', 'Traditional University')
    save_consent(name)
    field_encoded = field_mapping.get(field, 0)
    institution_encoded = institution_mapping.get(institution, 0)
    input_data = np.array([[nqf_level, duration, skills_count, field_encoded, institution_encoded]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    confidence = round(max(probabilities) * 100, 2)
    save_prediction(nqf_level, duration, skills_count, field, institution, prediction, f'{confidence}%')

    career_map = {
        'STEM': ['Data Scientist', 'Software Engineer', 'Mathematician', 'Statistician'],
        'Health': ['Medical Doctor', 'Nurse', 'Pharmacist', 'Physiotherapist'],
        'Engineering': ['Civil Engineer', 'Electrical Engineer', 'Mechanical Engineer'],
        'Business': ['Accountant', 'Business Analyst', 'Economist', 'Financial Manager'],
        'Arts': ['Graphic Designer', 'Journalist', 'Social Worker', 'Teacher'],
        'Education': ['Teacher', 'Educational Psychologist', 'Lecturer', 'Curriculum Developer']
    }

    university_map = {
        'STEM': ['University of Zululand', 'UKZN', 'Wits University'],
        'Health': ['UKZN Medical School', 'University of Pretoria', 'Stellenbosch University'],
        'Engineering': ['Durban University of Technology', 'UKZN', 'Cape Peninsula University of Technology'],
        'Business': ['University of Zululand', 'UKZN', 'University of Johannesburg'],
        'Arts': ['University of Zululand', 'UKZN', 'University of Cape Town'],
        'Education': ['University of Zululand', 'UKZN', 'University of South Africa (UNISA)']
    }

    skills_map = {
        'STEM': ['Logic', 'Programming', 'Mathematics', 'Coding', 'Advanced Physics', 'Calculus'],
        'Health': ['Diagnosis', 'Science', 'Empathy', 'Chemistry', 'Patient Care'],
        'Engineering': ['Physics', 'Mathematics', 'Hands-on work', 'Geology', 'Circuitry'],
        'Business': ['Numeracy', 'Ethics', 'Analysis', 'Communication', 'Sales'],
        'Arts': ['Creativity', 'Design', 'Studio Work', 'Sewing'],
        'Education': ['Patience', 'Literacy', 'Child Psychology', 'Communication']
    }

    careers = career_map.get(field, ['Please consult a career counsellor'])
    universities = university_map.get(field, ['Please consult your institution'])
    skills = skills_map.get(field, ['Communication', 'Critical Thinking', 'Problem Solving'])
    reasoning = f"You selected {field} as your field of study. Based on your NQF level {nqf_level} and {skills_count} subjects, the system recommends careers in {field}. Your top recommended career is {careers[0]} which requires skills like {', '.join(skills[:3])}. Your bursary availability is predicted to be {prediction} with {confidence}% confidence."

    return jsonify({
        'prediction': prediction,
        'confidence': f'{confidence}%',
        'message': f'Bursary availability is predicted to be {prediction}',
        'careers': careers,
        'universities': universities,
        'skills': skills,
        'reasoning': reasoning
    })

@app.route('/chat', methods=['POST'])
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        messages = data.get('messages', [])
        if not messages:
            return jsonify({'reply': 'Hello! I am your Career Guidance Assistant. Ask me anything about careers, bursaries or universities!'})

        last_message = messages[-1]['content'].lower()

        if any(w in last_message for w in ['nsfas', 'bursary', 'bursaries', 'funding', 'isfap', 'financial']):
            reply = """Here are the main bursaries for South African students:

**NSFAS** — For households earning less than R350,000/year. Covers tuition, accommodation and meals. Apply at nsfas.org.za

**ISFAP** — For missing middle students. Full cost of study covered. Apply at isfap.org.za

**CSIR** — For STEM students. Tuition plus monthly stipend.

**Funza Lushaka** — For teaching students. Full bursary plus R7,000/month allowance.

**Anglo American** — For Engineering and Mining students.

Apply early — most bursaries open in March for the following year!"""

        elif any(w in last_message for w in ['engineering', 'civil', 'electrical', 'mechanical']):
            reply = """Engineering is an excellent career in South Africa!

**Popular careers:** Civil Engineer, Electrical Engineer, Mechanical Engineer, Chemical Engineer

**Requirements:** Mathematics and Physical Sciences at 60%+

**Best universities:** DUT, UKZN, Wits, UCT, UP

**Bursaries:** CSIR, Anglo American, Eskom, Sasol

**Starting salary:** R350,000 - R500,000/year"""

        elif any(w in last_message for w in ['medicine', 'doctor', 'mbchb', 'health', 'nursing', 'nurse', 'pharmacy']):
            reply = """Healthcare is one of the most rewarding careers!

**Popular careers:** Medical Doctor (MBChB - 6 years), Nurse (4 years), Pharmacist (4 years), Physiotherapist (4 years)

**Requirements for Medicine:** Mathematics and Life Sciences at 70%+. APS score of 36+

**Best universities:** UKZN Medical School, UP, UCT, Wits, Stellenbosch

**Bursaries:** Department of Health bursary, NSFAS"""

        elif any(w in last_message for w in ['computer', 'software', 'it ', 'programming', 'coding', 'data science', 'stem', 'technology']):
            reply = """Technology careers are in high demand in South Africa!

**Popular careers:** Software Engineer, Data Scientist, Cybersecurity Analyst, IT Support

**Requirements:** Mathematics at 50%+

**Best universities:** UNIZULU, UKZN, Wits, UCT, CPUT

**Bursaries:** CSIR, Vodacom, MTN, Standard Bank

South Africa has a huge shortage of IT professionals — excellent job opportunities!"""

        elif any(w in last_message for w in ['business', 'accounting', 'finance', 'economics', 'bcom']):
            reply = """Business and Finance careers offer great opportunities!

**Popular careers:** Chartered Accountant (CA), Business Analyst, Financial Manager, Economist

**Requirements:** Mathematics at 50%+. Accounting is very helpful.

**Best universities:** UNIZULU, UKZN, UJ, UP, Stellenbosch

**Bursaries:** SAICA, major banks, NSFAS

CA(SA) is one of the highest paid qualifications in South Africa!"""

        elif any(w in last_message for w in ['teaching', 'teacher', 'education', 'bed', 'pgce']):
            reply = """Teaching is one of the most impactful careers!

**Qualifications:** BEd (4 years) or PGCE (1 year after a degree)

**Requirements:** Any matric subjects depending on what you want to teach

**Best universities:** UNIZULU, UKZN, UNISA, UP

**Bursaries:** Funza Lushaka — full tuition plus R7,000/month!

South Africa needs good teachers urgently — great job security!"""

        elif any(w in last_message for w in ['ukzn', 'unizulu', 'dut', 'university', 'universities', 'college', 'apply', 'application']):
            reply = """Key South African universities:

**UNIZULU (University of Zululand)** — KwaZulu-Natal. Strong in Education, Arts, Science. unizulu.ac.za

**UKZN** — Multiple KZN campuses. Strong in Medicine, Engineering, Law. ukzn.ac.za

**DUT (Durban University of Technology)** — Strong in Engineering, IT, Business. dut.ac.za

**Tips:** Apply before September. Apply to at least 3 universities. Apply for NSFAS at the same time!"""

        elif any(w in last_message for w in ['matric', 'grade 12', 'aps', 'marks', 'subjects', 'subject']):
            reply = """Matric and APS advice:

**APS Score calculation** (6 best subjects):
- 80-100% = 7 points
- 70-79% = 6 points  
- 60-69% = 5 points
- 50-59% = 4 points
- 40-49% = 3 points

**Minimum APS:** Medicine 36+ | Engineering 30+ | Teaching 24+ | Business 26+

**Important:** Mathematics (not Maths Literacy) opens far more career doors!"""

        elif any(w in last_message for w in ['salary', 'earn', 'money', 'paid', 'income']):
            reply = """Typical starting salaries in South Africa:

**High earning:**
- Medical Doctor: R600,000 - R1,200,000/year
- Chartered Accountant: R500,000 - R800,000/year
- Software Engineer: R400,000 - R700,000/year
- Civil Engineer: R350,000 - R600,000/year

**Mid range:**
- Teacher: R200,000 - R350,000/year
- Nurse: R180,000 - R300,000/year

Salary grows with experience. Choose work you enjoy — success follows passion!"""

        elif any(w in last_message for w in ['hello', 'hi', 'hey', 'sawubona', 'good morning', 'good afternoon']):
            reply = """Hello! Welcome to the Career Guidance Assistant!

I can help you with:
- Career recommendations based on your subjects
- Bursary information (NSFAS, ISFAP, Funza Lushaka)
- University information and applications
- Matric and APS score advice
- Salary expectations

What would you like to know? Just ask!"""

        elif any(w in last_message for w in ['mathematics', 'maths', 'math']):
            reply = """Mathematics opens many doors in South Africa!

**Careers with Mathematics:**
- Engineering (Civil, Electrical, Mechanical)
- Data Science and Software Engineering
- Accounting and Finance (CA, CFO)
- Medicine and Pharmacy
- Teaching (Maths teacher — very high demand!)
- Architecture

**Key point:** Always take Pure Mathematics over Mathematical Literacy if possible. It dramatically increases your career options and university admission chances!"""

        elif any(w in last_message for w in ['life sciences', 'biology']):
            reply = """Life Sciences leads to many exciting careers!

**Careers with Life Sciences:**
- Medical Doctor
- Nurse or Physiotherapist
- Pharmacist
- Environmental Scientist
- Biotechnologist
- Teacher (Life Sciences)

**Best combination:** Life Sciences + Mathematics + Physical Sciences gives you access to Medicine, Pharmacy and all health careers.

**Universities:** UKZN Medical School, UP, Stellenbosch, UNIZULU"""

        else:
            reply = """Thank you for your question! I am your South African Career Guidance Assistant.

I can help you with:
- **Careers** — Engineering, Medicine, IT, Teaching, Business
- **Bursaries** — NSFAS, ISFAP, Funza Lushaka, CSIR
- **Universities** — UNIZULU, UKZN, DUT and others
- **Subjects** — matric requirements and APS scores
- **Salaries** — what you can expect to earn

Try asking:
- "What careers suit Mathematics?"
- "How do I apply for NSFAS?"
- "What APS do I need for Engineering?"
- "Tell me about UKZN"

I am here to help! 😊"""

        return jsonify({'reply': reply})

    except Exception as e:
        return jsonify({'reply': 'Error: ' + str(e)})