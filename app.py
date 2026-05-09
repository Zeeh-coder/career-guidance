from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
import joblib
import numpy as np
import os
import requests
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
        api_key = os.environ.get('GEMINI_API_KEY', '')

        if not api_key:
            return jsonify({'reply': 'API key is missing. Please set GEMINI_API_KEY on Render.'})

        gemini_messages = []
        for msg in messages:
            role = 'user' if msg['role'] == 'user' else 'model'
            gemini_messages.append({
                'role': role,
                'parts': [{'text': msg['content']}]
            })

        response = requests.post(
            f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}',
            headers={'Content-Type': 'application/json'},
            json={
                'system_instruction': {
                    'parts': [{'text': 'You are an AI Career Guidance Assistant for South African high school and university students. Help them explore career pathways, university requirements, subject choices, bursaries like NSFAS and ISFAP, and study tips. Be warm, encouraging and practical. Focus on the South African context including NQF levels, matric requirements, UKZN, UNIZULU, DUT and other SA institutions.'}]
                },
                'contents': gemini_messages
            },
            timeout=30
        )

        result = response.json()

        if 'candidates' in result:
            reply = result['candidates'][0]['content']['parts'][0]['text']
            return jsonify({'reply': reply})
        elif 'error' in result:
            return jsonify({'reply': 'API error: ' + str(result['error'].get('message', 'Unknown error'))})
        else:
            return jsonify({'reply': 'Unexpected response. Please try again.'})

    except requests.exceptions.Timeout:
        return jsonify({'reply': 'The AI took too long to respond. Please try again.'})
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