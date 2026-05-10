from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
import joblib
import numpy as np
import os
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
def chat():
    try:
        data = request.json
        messages = data.get('messages', [])
        if not messages:
            return jsonify({'reply': 'Hello! I am your Career Guidance Assistant. Ask me anything about careers, bursaries or universities!'})

        last_message = messages[-1]['content'].lower()

        if any(w in last_message for w in ['nsfas', 'bursary', 'bursaries', 'funding', 'isfap', 'financial']):
            reply = "NSFAS - For households earning less than R350,000/year. Apply at nsfas.org.za\n\nISFAP - For missing middle students. Full cost of study. Apply at isfap.org.za\n\nCSIR - For STEM students. Tuition plus monthly stipend.\n\nFunza Lushaka - For teaching students. Full bursary plus R7,000/month.\n\nAnglo American - For Engineering and Mining students.\n\nApply early - most bursaries open in March!"

        elif any(w in last_message for w in ['engineering', 'civil', 'electrical', 'mechanical']):
            reply = "Engineering careers in South Africa:\n\nCivil Engineer, Electrical Engineer, Mechanical Engineer, Chemical Engineer\n\nRequirements: Mathematics and Physical Sciences at 60%+\n\nBest universities: DUT, UKZN, Wits, UCT, UP\n\nBursaries: CSIR, Anglo American, Eskom, Sasol\n\nStarting salary: R350,000 - R500,000/year"

        elif any(w in last_message for w in ['medicine', 'doctor', 'mbchb', 'health', 'nursing', 'nurse', 'pharmacy']):
            reply = "Healthcare careers:\n\nMedical Doctor (MBChB - 6 years), Nurse (4 years), Pharmacist (4 years), Physiotherapist (4 years)\n\nRequirements for Medicine: Mathematics and Life Sciences at 70%+. APS score of 36+\n\nBest universities: UKZN Medical School, UP, UCT, Wits, Stellenbosch\n\nBursaries: Department of Health bursary, NSFAS"

        elif any(w in last_message for w in ['computer', 'software', 'programming', 'coding', 'data', 'stem', 'technology', 'it']):
            reply = "Technology careers in high demand:\n\nSoftware Engineer, Data Scientist, Cybersecurity Analyst, IT Support\n\nRequirements: Mathematics at 50%+\n\nBest universities: UNIZULU, UKZN, Wits, UCT, CPUT\n\nBursaries: CSIR, Vodacom, MTN, Standard Bank\n\nSouth Africa has a huge shortage of IT professionals!"

        elif any(w in last_message for w in ['business', 'accounting', 'finance', 'economics', 'bcom']):
            reply = "Business and Finance careers:\n\nChartered Accountant (CA), Business Analyst, Financial Manager, Economist\n\nRequirements: Mathematics at 50%+\n\nBest universities: UNIZULU, UKZN, UJ, UP, Stellenbosch\n\nBursaries: SAICA, major banks, NSFAS\n\nCA(SA) is one of the highest paid qualifications in South Africa!"

        elif any(w in last_message for w in ['teaching', 'teacher', 'education', 'bed', 'pgce']):
            reply = "Teaching careers:\n\nBEd (4 years) or PGCE (1 year after a degree)\n\nBest universities: UNIZULU, UKZN, UNISA, UP\n\nBursaries: Funza Lushaka - full tuition plus R7,000/month!\n\nSouth Africa needs good teachers urgently - great job security!"

        elif any(w in last_message for w in ['ukzn', 'unizulu', 'dut', 'university', 'universities', 'college', 'apply']):
            reply = "Key South African universities:\n\nUNIZULU - KwaZulu-Natal. Strong in Education, Arts, Science. unizulu.ac.za\n\nUKZN - Multiple KZN campuses. Strong in Medicine, Engineering, Law. ukzn.ac.za\n\nDUT - Strong in Engineering, IT, Business. dut.ac.za\n\nTip: Apply before September. Apply to at least 3 universities!"

        elif any(w in last_message for w in ['matric', 'grade 12', 'aps', 'marks', 'subjects']):
            reply = "APS Score (6 best subjects):\n80-100% = 7 points\n70-79% = 6 points\n60-69% = 5 points\n50-59% = 4 points\n40-49% = 3 points\n\nMinimum APS: Medicine 36+ | Engineering 30+ | Teaching 24+ | Business 26+\n\nAlways take Pure Mathematics over Mathematical Literacy!"

        elif any(w in last_message for w in ['salary', 'earn', 'money', 'paid', 'income']):
            reply = "Starting salaries in South Africa:\n\nMedical Doctor: R600,000 - R1,200,000/year\nChartered Accountant: R500,000 - R800,000/year\nSoftware Engineer: R400,000 - R700,000/year\nCivil Engineer: R350,000 - R600,000/year\nTeacher: R200,000 - R350,000/year\nNurse: R180,000 - R300,000/year"

        elif any(w in last_message for w in ['mathematics', 'maths', 'math']):
            reply = "Mathematics opens many doors:\n\nEngineering, Data Science, Software Engineering, Accounting, Finance, Medicine, Pharmacy, Teaching (Maths teacher - very high demand!), Architecture\n\nAlways take Pure Mathematics over Mathematical Literacy if possible!"

        elif any(w in last_message for w in ['hello', 'hi', 'hey', 'sawubona']):
            reply = "Hello! Welcome to the Career Guidance Assistant!\n\nI can help you with:\n- Career recommendations\n- Bursary information (NSFAS, ISFAP, Funza Lushaka)\n- University information\n- Matric and APS advice\n- Salary expectations\n\nWhat would you like to know?"

        else:
            reply = "I am your South African Career Guidance Assistant.\n\nI can help you with:\n- Careers - Engineering, Medicine, IT, Teaching, Business\n- Bursaries - NSFAS, ISFAP, Funza Lushaka, CSIR\n- Universities - UNIZULU, UKZN, DUT\n- Subjects and APS scores\n- Salaries\n\nTry asking: What careers suit Mathematics? or How do I apply for NSFAS?"

        return jsonify({'reply': reply})

    except Exception as e:
        return jsonify({'reply': 'Error: ' + str(e)})

@app.route('/predictions', methods=['GET'])
def predictions():
    rows = get_all_predictions()
    result = []
    for row in rows:
        result.append({
            'id': row[0],
            'nqf_level': row[1],
            'duration': row[2],
            'skills_count': row[3],
            'field': row[4],
            'institution': row[5],
            'prediction': row[6],
            'confidence': row[7],
            'timestamp': row[8]
        })
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)