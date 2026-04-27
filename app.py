import os, sys, glob
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, \
logout_user, current_user
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
import torch
from PIL import Image

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ai-detector-secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    analyses = db.relationship('AnalysisHistory', backref='user', lazy=True)

class AnalysisHistory(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path   = db.Column(db.String(255), nullable=False)
    label        = db.Column(db.String(100), nullable=False)
    confidence   = db.Column(db.String(20), nullable=False)
    is_ai        = db.Column(db.Boolean, nullable=False)
    analysed_at  = db.Column(db.DateTime, default=db.func.now())

class ContactMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    company = db.Column(db.String(100), nullable=True)
    email = db.Column(db.String(120), nullable=False)
    subject = db.Column(db.String(50), nullable=True)
    message = db.Column(db.Text, nullable=False)
    submitted_at = db.Column(db.DateTime, default=db.func.now())

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model import get_model
from custom_dataset import get_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_ai_model():
    m = get_model(device)
    pref = 'models/best_model.pth'
    candidates = glob.glob('models/*.pth')
    if os.path.exists(pref):
        weights = pref
    elif candidates:
        weights = candidates[0]
    else:
        return None, get_transform()
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    return m, get_transform()

AI_MODEL, AI_TRANSFORM = load_ai_model()

# ── Static / Info Routes ─────────────────────────────────────────────────────
@app.route('/')
def index(): return render_template('index.html')

@app.route('/faq')
def faq(): return render_template('faq.html')

@app.route('/about')
def about(): return render_template('about.html')

@app.route('/terms')
def terms(): return render_template('terms.html')

@app.route('/privacy')
def privacy(): return render_template('privacy.html')

@app.route('/api-docs')
@login_required
def api_docs(): return render_template('api_docs.html')

# ── Contact ──────────────────────────────────────────────────────────────────
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name    = request.form.get('name', '') or request.form.get('company', '')
        company = request.form.get('company', '')
        email   = request.form.get('email', '')
        subject = request.form.get('subject', '')
        message = request.form.get('message', '')
        if name and company and email and message:
            new_msg = ContactMessage(
                name=name,
                company=company,
                email=email,
                subject=subject,
                message=message
            )
            try:
                db.session.add(new_msg)
                db.session.commit()
                flash(f"Thanks {name}! We've received your message and will reply shortly.", 'success')
            except Exception as e:
                print("CONTACT FORM ERROR:", e)
                db.session.rollback()
                flash('An error occurred. Please try again.', 'error')
        else:
            flash('Please fill in all required fields.', 'error')
        return redirect(url_for('contact'))
    return render_template('contact.html')

# ── Auth ─────────────────────────────────────────────────────────────────────
@app.route('/signup')
def signup():
    """Legacy route — redirect to register."""
    return redirect(url_for('register'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        name             = request.form.get('name', '')
        email            = request.form['email']
        password         = request.form['password']
        confirm_password = request.form['confirm_password']

        if not name or len(name.strip()) < 2:
            flash('Name must be at least 2 characters long.', 'error')
            return redirect(url_for('register'))
        if not email or '@' not in email:
            flash('Please enter a valid email address.', 'error')
            return redirect(url_for('register'))
        if (len(password) < 8
                or not any(c.isdigit() for c in password)
                or not any(c.isalpha() for c in password)
                or not any(not c.isalnum() for c in password)):
            flash('Password must be at least 8 characters and include letters, numbers, and special characters.', 'error')
            return redirect(url_for('register'))
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email.strip()).first():
            flash('Email already registered. Please log in.', 'error')
            return redirect(url_for('register'))

        pw = bcrypt.generate_password_hash(password).decode('utf-8')
        import uuid
        username = (email.strip().split('@')[0][:10] + '_' + uuid.uuid4().hex[:8]).lower()
        new_user = User(name=name.strip(), username=username, email=email.strip(), password=pw)
        try:
            db.session.add(new_user); db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('index'))
        except Exception as e:
            print("REGISTRATION ERROR:", e)
            db.session.rollback()
            flash('An error occurred during registration. Please try again.', 'error')
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()
        if user:
            try:
                if bcrypt.check_password_hash(user.password, request.form['password']):
                    login_user(user)
                    return redirect(url_for('index'))
            except ValueError:
                pass
        flash('Invalid email or password.', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

@app.route('/update_name', methods=['POST'])
@login_required
def update_name():
    new_name = request.form.get('name', '').strip()
    if new_name and len(new_name) >= 2:
        current_user.name = new_name
        db.session.commit()
        flash('Name updated successfully.', 'success')
    else:
        flash('Name must be at least 2 characters long.', 'error')
    return redirect(url_for('profile'))

# ── Predict (Web UI) ─────────────────────────────────────────────────────────
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    res = None
    if request.method == 'POST':
        if AI_MODEL is None:
            flash('Model weights not found. Please train the model first.', 'error')
            return render_template('predict.html', result=None)
        f = request.files.get('file')
        if not f or f.filename == '':
            flash('No file selected.', 'error')
            return render_template('predict.html', result=None)
        path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(path)
        img = AI_TRANSFORM(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            out = AI_MODEL(img)
            _, pred = out.max(1)
        conf = torch.nn.functional.softmax(out, dim=1)[0][pred.item()].item()
        label = "AI-Generated (Fake)" if pred.item() == 0 else "Real"
        conf_str = f"{conf*100:.2f}%"
        # ── Save to history ──
        record = AnalysisHistory(
            user_id    = current_user.id,
            image_path = secure_filename(f.filename),
            label      = label,
            confidence = conf_str,
            is_ai      = (pred.item() == 0)
        )
        db.session.add(record)
        db.session.commit()
        res = {
            'class': label,
            'confidence': conf_str,
            'image_path': secure_filename(f.filename)
        }
    
    # ── Fetch recent history ──
    all_recent = AnalysisHistory.query.filter_by(user_id=current_user.id)\
                 .order_by(AnalysisHistory.analysed_at.desc()).all()
    recent = []
    dirty = False
    for r in all_recent:
        if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], r.image_path)):
            if len(recent) < 3:
                recent.append(r)
        else:
            db.session.delete(r)
            dirty = True
    if dirty:
        db.session.commit()
             
    return render_template('predict.html', result=res, recent_history=recent)

# ── History ──────────────────────────────────────────────────────────────────
@app.route('/history')
@login_required
def history():
    records = AnalysisHistory.query.filter_by(user_id=current_user.id)\
                .order_by(AnalysisHistory.analysed_at.desc()).all()
    
    valid_records = []
    dirty = False
    for r in records:
        if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], r.image_path)):
            valid_records.append(r)
        else:
            db.session.delete(r)
            dirty = True
    if dirty:
        db.session.commit()

    return render_template('history.html', records=valid_records)

# ── API Endpoint ─────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for Swiggy/Zomato backend integration."""
    if AI_MODEL is None:
        return jsonify({'error': 'Model not loaded — weights missing on server'}), 503
    f = request.files.get('file')
    if not f or f.filename == '':
        return jsonify({'error': 'No file part or empty filename'}), 400
    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
    f.save(path)
    try:
        img = AI_TRANSFORM(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            out = AI_MODEL(img)
            _, pred = out.max(1)
        conf = torch.nn.functional.softmax(out, dim=1)[0][pred.item()].item()
        label = "AI Generated" if pred.item() == 0 else "Real"
        is_fraud = pred.item() == 0
        return jsonify({
            'class': label,
            'is_fraud': is_fraud,
            'confidence': round(conf * 100, 2),
            'confidence_display': f"{conf*100:.2f}%",
            'filename': secure_filename(f.filename),
            'recommendation': 'REJECT_CLAIM' if is_fraud else 'APPROVE_CLAIM'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context(): db.create_all()
    app.run(debug=True)
