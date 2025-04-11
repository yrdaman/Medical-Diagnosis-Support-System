# 🩺 Medical Diagnosis Support System

An AI-powered web application designed to assist users in identifying possible diseases based on their symptoms using Natural Language Processing (NLP), Machine Learning (ML), and pre-trained medical models. The system supports both **text and voice-based** symptom input and provides **first-aid suggestions** and **precautions**, improving accessibility and early awareness in healthcare.

---

## 🔍 Features

- ✅ Symptom-based disease prediction using Random Forest Classifier  
- ✅ Multi-symptom tag input system with auto-suggestions  
- ✅ Speech-to-text symptom input using Whisper model  
- ✅ Auto-correction and normalization using Gemini AI  
- ✅ Medical NER using `en_ner_bc5cdr_md` (SciSpacy pre-trained model)  
- ✅ Description and precautions for predicted diseases  
- ✅ User feedback and learning system (CSV-based logging)  
- ✅ User authentication system (Sign up/Login)  
- ✅ Clean, responsive interface with tag-based selection  

---

## 🧠 Technologies Used

### 🔧 Backend
- Python (Django)
- Machine Learning (Scikit-learn)
- Pre-trained models (`en_ner_bc5cdr_md`, Whisper, Gemini API)
- Joblib for model serialization
- FFmpeg for audio processing

### 🎨 Frontend
- HTML5 / CSS3
- JavaScript (minimal, for interactivity)
- Responsive design (optimized layout and aesthetics)

### 🧪 Data
- `updated_dataset.csv`: Cleaned disease-symptom mapping  
- `symptom_Description.csv`: Disease-wise descriptions  
- `symptom_precaution.csv`: Disease-wise precautionary measures  
- `feedback.csv`: Captures user corrections to improve model  

---

## 🗂️ Project Structure

healthcare_assistant/
│
├── assistant/
│   ├── migrations/
│   ├── templates/
│   │   └── assistant/
│   │       ├── index.html
│   │       └── results.html
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── urls.py
│   ├── views.py
│   ├── disease_predictor.py
│   ├── nlp_processor.py
│   └── dataset/
│       ├── updated_dataset.csv
│       ├── symptom_Description.csv
│       ├── symptom_precaution.csv
│       └── feedback.csv
│
├── saved_models/
│   ├── disease_model.pkl
│   └── mlb.pkl
│
├── manage.py
├── requirements.txt
└── README.md


**Use Cases**

Self-diagnosis tool for early awareness
Educational demo for AI in healthcare
Low-resource telemedicine aid
