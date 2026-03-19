import os
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///repository.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'super-secret-production-key-change-me'
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB max upload

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# --- Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False) # 'admin', 'faculty', 'student'

class Faculty(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    bio = db.Column(db.Text)
    research_interests = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

class Resource(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    type = db.Column(db.String(50), nullable=False) # 'course' or 'publication'
    filename = db.Column(db.String(200), nullable=False)
    faculty_id = db.Column(db.Integer, db.ForeignKey('faculty.id'), nullable=False)

# --- Routes: Auth ---
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    new_user = User(username=data['username'], password=hashed_password, role=data.get('role', 'student'))
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "User registered successfully"}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data['username']).first()
    if user and bcrypt.check_password_hash(user.password, data['password']):
        access_token = create_access_token(identity={'id': user.id, 'role': user.role})
        return jsonify(access_token=access_token, role=user.role), 200
    return jsonify({"message": "Invalid credentials"}), 401

# --- Routes: Analytics (Admin) ---
@app.route('/api/analytics', methods=['GET'])
@jwt_required()
def analytics():
    user = get_jwt_identity()
    if user['role'] != 'admin':
        return jsonify({"message": "Unauthorized"}), 403
    
    return jsonify({
        "total_faculty": Faculty.query.count(),
        "total_courses": Resource.query.filter_by(type='course').count(),
        "total_publications": Resource.query.filter_by(type='publication').count()
    }), 200

# --- Routes: File Upload (Faculty/Admin) ---
@app.route('/api/upload', methods=['POST'])
@jwt_required()
def upload_file():
    user = get_jwt_identity()
    if user['role'] not in ['admin', 'faculty']:
        return jsonify({"message": "Unauthorized"}), 403

    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # Save metadata to DB
    new_resource = Resource(
        title=request.form['title'],
        description=request.form['description'],
        type=request.form['type'],
        filename=filename,
        faculty_id=request.form['faculty_id']
    )
    db.session.add(new_resource)
    db.session.commit()

    return jsonify({"message": "File uploaded successfully"}), 201

# --- Routes: AI Search ---
@app.route('/api/search', methods=['GET'])
@jwt_required()
def search():
    query = request.args.get('q', '')
    if not query:
        return jsonify([])

    resources = Resource.query.all()
    if not resources:
        return jsonify([])

    # Prepare corpus for TF-IDF
    documents = [f"{r.title} {r.description}" for r in resources]
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Transform query and compute similarity
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top 5 results
    top_indices = similarities.argsort()[-5:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.05:  # Relevance threshold
            res = resources[idx]
            results.append({
                "id": res.id,
                "title": res.title,
                "description": res.description[:100] + "...",
                "type": res.type,
                "download_url": f"/api/download/{res.filename}",
                "score": float(similarities[idx])
            })
            
    return jsonify(results), 200

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Create default admin if not exists
        if not User.query.filter_by(username='admin').first():
            hashed_pw = bcrypt.generate_password_hash('admin123').decode('utf-8')
            admin = User(username='admin', password=hashed_pw, role='admin')
            db.session.add(admin)
            db.session.commit()
    app.run(host='0.0.0.0', port=5000, debug=True)
