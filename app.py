import os
import json
from datetime import timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///repository.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'super-secret-production-key-change-me')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=1)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# ─── MODELS ──────────────────────────────────────────────────────────────────
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    full_name = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(20), nullable=False) # admin, faculty, student
    department = db.Column(db.String(100)) # relevant for faculty
    bio = db.Column(db.Text)
    research_interests = db.Column(db.Text)

class Resource(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    type = db.Column(db.String(50), nullable=False) # course, publication
    filename = db.Column(db.String(200), nullable=False)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    uploader = db.relationship('User', backref='resources')

# ─── INIT DB & SEED DATA ─────────────────────────────────────────────────────
def init_db():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            db.session.add(User(username='admin', password=bcrypt.generate_password_hash('admin123').decode('utf-8'), full_name='System Admin', role='admin'))
            db.session.add(User(username='faculty1', password=bcrypt.generate_password_hash('fac123').decode('utf-8'), full_name='Dr. Ibrahim', role='faculty', department='Computer Science', research_interests='AI, Machine Learning'))
            db.session.add(User(username='student1', password=bcrypt.generate_password_hash('stu123').decode('utf-8'), full_name='Amina Yusuf', role='student'))
            db.session.commit()

# ─── AUTHENTICATION ──────────────────────────────────────────────────────────
@app.route('/api/auth/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS': return jsonify({}), 200
    data = request.json
    user = User.query.filter_by(username=data.get('username')).first()
    if user and bcrypt.check_password_hash(user.password, data.get('password')):
        token = create_access_token(identity={'id': user.id, 'role': user.role, 'name': user.full_name})
        return jsonify({
            'success': True,
            'token': token,
            'user': {'id': user.id, 'username': user.username, 'full_name': user.full_name, 'role': user.role, 'department': user.department}
        })
    return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

# ─── ANALYTICS (ADMIN) ───────────────────────────────────────────────────────
@app.route('/api/analytics', methods=['GET'])
@jwt_required()
def analytics():
    claims = get_jwt_identity()
    if claims['role'] != 'admin': return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    return jsonify({
        'success': True,
        'total_faculty': User.query.filter_by(role='faculty').count(),
        'total_students': User.query.filter_by(role='student').count(),
        'total_courses': Resource.query.filter_by(type='course').count(),
        'total_publications': Resource.query.filter_by(type='publication').count(),
        'recent_uploads': [{'title': r.title, 'type': r.type, 'uploader': r.uploader.full_name} for r in Resource.query.order_by(Resource.created_at.desc()).limit(5)]
    })

# ─── FACULTY MANAGEMENT (ADMIN) ──────────────────────────────────────────────
@app.route('/api/faculty', methods=['GET', 'POST'])
@jwt_required()
def manage_faculty():
    claims = get_jwt_identity()
    if request.method == 'GET':
        faculties = User.query.filter_by(role='faculty').all()
        return jsonify([{'id': f.id, 'full_name': f.full_name, 'username': f.username, 'department': f.department, 'research_interests': f.research_interests} for f in faculties])
    
    if claims['role'] != 'admin': return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    data = request.json
    new_fac = User(
        username=data['username'],
        password=bcrypt.generate_password_hash(data['password']).decode('utf-8'),
        full_name=data['full_name'],
        department=data['department'],
        research_interests=data.get('research_interests', ''),
        role='faculty'
    )
    db.session.add(new_fac)
    db.session.commit()
    return jsonify({'success': True})

# ─── RESOURCES & FILE UPLOADS ────────────────────────────────────────────────
@app.route('/api/resources', methods=['GET', 'POST'])
@jwt_required()
def manage_resources():
    claims = get_jwt_identity()
    
    if request.method == 'GET':
        query = Resource.query
        # Faculty only sees their own by default unless searching globally
        if claims['role'] == 'faculty' and request.args.get('all') != 'true':
            query = query.filter_by(uploaded_by=claims['id'])
        resources = query.order_by(Resource.created_at.desc()).all()
        return jsonify([{
            'id': r.id, 'title': r.title, 'description': r.description, 
            'type': r.type, 'filename': r.filename, 'uploader': r.uploader.full_name,
            'date': r.created_at.strftime("%Y-%m-%d")
        } for r in resources])

    # POST (Upload)
    if claims['role'] not in ['admin', 'faculty']: return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    if 'file' not in request.files: return jsonify({'success': False, 'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '': return jsonify({'success': False, 'error': 'No selected file'}), 400

    filename = secure_filename(f"{claims['id']}_{file.filename}")
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    new_resource = Resource(
        title=request.form['title'],
        description=request.form['description'],
        type=request.form['type'],
        filename=filename,
        uploaded_by=claims['id']
    )
    db.session.add(new_resource)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ─── AI SEMANTIC SEARCH ──────────────────────────────────────────────────────
@app.route('/api/search', methods=['GET'])
@jwt_required()
def ai_search():
    query = request.args.get('q', '')
    if not query: return jsonify([])

    resources = Resource.query.all()
    if not resources: return jsonify([])

    # Create corpus including title, description, and faculty metadata
    documents = [f"{r.title} {r.description} {r.type} {r.uploader.full_name} {r.uploader.research_interests}" for r in resources]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Top 10 results over a minimal threshold
    top_indices = similarities.argsort()[-10:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.02:
            res = resources[idx]
            results.append({
                'id': res.id,
                'title': res.title,
                'description': res.description,
                'type': res.type,
                'uploader': res.uploader.full_name,
                'filename': res.filename,
                'score': round(float(similarities[idx]) * 100, 1),
                'date': res.created_at.strftime("%Y-%m-%d")
            })
            
    return jsonify(results)

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
                   
