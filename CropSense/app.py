import numpy as np
import pickle
import sys
import logging
import os
import csv  
import datetime
from flask import Flask, request, render_template, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from sklearn.exceptions import InconsistentVersionWarning
import warnings
from flask import jsonify
sys.stdout.reconfigure(encoding='utf-8')
warnings.simplefilter("ignore", InconsistentVersionWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
crop_details = {
    "cotton": {
        "type": "Staple Crop",
        "advantages": "Highly profitable in dry areas; supports textile industries.",
        "fertilizers": "Nitrogen, Phosphorus, Potassium"
    },
    "rice": {
        "type": "Cereal Crop",
        "advantages": "Primary food grain in Asia; high yield.",
        "fertilizers": "Urea, DAP, MOP"
    },
    "maize": {
        "type": "Cereal Crop",
        "advantages": "Used for food, fodder, and industrial products.",
        "fertilizers": "Nitrogen-rich fertilizers"
    },
    "coffee": {
        "type": "Commercial Crop",
        "advantages": "High demand globally; profitable in hilly regions.",
        "fertilizers": "NPK, compost, bone meal"
    },
    "chickpea": {
        "type": "Pulse Crop",
        "advantages": "Improves soil fertility; protein-rich food.",
        "fertilizers": "Super phosphate, potash"
    },
    "pigeonpea": {
        "type": "Pulse Crop",
        "advantages": "Drought-resistant and restores soil nitrogen.",
        "fertilizers": "DAP, SSP"
    },
    "mango": {
        "type": "Fruit Crop",
        "advantages": "High economic value; widely loved tropical fruit.",
        "fertilizers": "Farmyard manure, NPK"
    },
    "orange": {
        "type": "Fruit Crop",
        "advantages": "Rich in vitamin C; good commercial value.",
        "fertilizers": "Potassium sulfate, manure"
    },
    "papaya": {
        "type": "Fruit Crop",
        "advantages": "Fast-growing and rich in nutrients.",
        "fertilizers": "Urea, cow dung, compost"
    },
    "watermelon": {
        "type": "Fruit Crop",
        "advantages": "High water content; good for summer markets.",
        "fertilizers": "Nitrogen, potash, phosphorus"
    },
    "lentil": {
        "type": "Pulse Crop",
        "advantages": "Nitrogen-fixing and protein-rich.",
        "fertilizers": "DAP, urea"
    },
    "pomegranate": {
        "type": "Fruit Crop",
        "advantages": "Resilient crop; high juice content and market value.",
        "fertilizers": "Organic manure, NPK"
    },
    "banana": {
        "type": "Fruit Crop",
        "advantages": "High yielding and supports export industry.",
        "fertilizers": "Nitrogen, phosphorus, potassium"
    },
    "grapes": {
        "type": "Fruit Crop",
        "advantages": "Used in wine and juice industries; high profit.",
        "fertilizers": "Potash, zinc sulfate"
    },
    "apple": {
        "type": "Fruit Crop",
        "advantages": "Stored for long; valuable in cold climates.",
        "fertilizers": "Urea, FYM, potassium sulfate"
    },
    "jute": {
        "type": "Fiber Crop",
        "advantages": "Eco-friendly fiber source; used in bags & mats.",
        "fertilizers": "Nitrogen, phosphorus"
    },
    "mungbean": {
        "type": "Pulse Crop",
        "advantages": "Short duration; enriches soil nitrogen.",
        "fertilizers": "Phosphate fertilizers, compost"
    }
}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'  
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cropsense.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    recommendations = db.relationship('Recommendation', backref='user', lazy=True)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Recommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    city = db.Column(db.String(100))
    nitrogen = db.Column(db.Float, nullable=False)
    phosphorus = db.Column(db.Float, nullable=False)
    potassium = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    ph = db.Column(db.Float, nullable=False)
    rainfall = db.Column(db.Float, nullable=False)
    recommended_crop = db.Column(db.String(100), nullable=False)
    alternatives = db.Column(db.String(255))
    field_label = db.Column(db.String(100))
    
    def __repr__(self):
        return f'<Recommendation {self.recommended_crop} for {self.user.username}>'

class CropData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    N = db.Column(db.Float, nullable=False)
    P = db.Column(db.Float, nullable=False)
    K = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    ph = db.Column(db.Float, nullable=False)
    rainfall = db.Column(db.Float, nullable=False)
    label = db.Column(db.String(100), nullable=False)
    
    def __repr__(self):
        return f'<CropData {self.label}>'

class Location(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    city = db.Column(db.String(100), nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    rainfall = db.Column(db.Float, nullable=False)
    
    def __repr__(self):
        return f'<Location {self.city}>'

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Load models safely
try:
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model/pca.pkl", "rb") as f:
        pca = pickle.load(f)
    with open("model/rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    with open("model/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    logging.info("Models loaded successfully!")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier

    scaler = StandardScaler()
    pca = PCA(n_components=3)
    rf_model = RandomForestClassifier()
    label_encoder = LabelEncoder()

    logging.warning("⚠️ Using fallback models (not accurate predictions).")

# Initialize database and load data
# Replace the @app.before_first_request decorator and function with this:
def init_db():
    with app.app_context():
        db.create_all()
        
        # Import crop dataset if it doesn't exist already
        if CropData.query.count() == 0:
            try:
                with open('pro/dataset/data.csv', 'r') as f:
                    csv_reader = csv.DictReader(f)
                    for row in csv_reader:
                        crop_data = CropData(
                            N=float(row['N']),
                            P=float(row['P']),
                            K=float(row['K']),
                            temperature=float(row['temperature']),
                            humidity=float(row['humidity']),
                            ph=float(row['ph']),
                            rainfall=float(row['rainfall']),
                            label=row['label']
                        )
                        db.session.add(crop_data)
                logging.info("CSV data imported successfully!")
                db.session.commit()
            except Exception as e:
                logging.error(f"Error importing CSV data: {e}")
                db.session.rollback()

# Sign-Up Route (Initial Page)
@app.route('/')
def sign_up():
    return render_template('sign.html')

# Home route (Page after Sign Up)
@app.route('/home')
def home():
    form_data = {}
    show_result = False
    prediction = ""
    alternatives = ""
    error = ""
    
    # Check if we need to load a previous recommendation
    rec_id = request.args.get('load')
    if rec_id and 'user_id' in session:
        try:
            rec = Recommendation.query.filter_by(id=rec_id, user_id=session['user_id']).first()
            if rec:
                form_data = {
                    'N': rec.nitrogen,
                    'P': rec.phosphorus,
                    'K': rec.potassium,
                    'temperature': rec.temperature,
                    'humidity': rec.humidity,
                    'ph': rec.ph,
                    'rainfall': rec.rainfall,
                    'label': rec.field_label or ''
                }
                prediction = f"✅ Recommended Crop: {rec.recommended_crop}"
                alternatives = f"🔹 Alternative crops: {rec.alternatives}"
                show_result = True
        except Exception as e:
            logging.error(f"Error loading recommendation: {e}")
            error = "Failed to load the recommendation"
    
    # Get list of available cities for the location selector
    cities = [location.city for location in Location.query.all()]
    
    return render_template('index.html', 
                          show_result=show_result, 
                          form_data=form_data,
                          prediction=prediction,
                          alternatives=alternatives,
                          error=error,
                          cities=cities)

# History route
@app.route('/history')
def history():
    if 'user_id' not in session:
        flash("Please log in to view your history.")
        return redirect(url_for('sign_up'))
        
    user = User.query.get(session['user_id'])
    recommendations = Recommendation.query.filter_by(user_id=user.id).order_by(Recommendation.date.desc()).all()
    
    return render_template('history.html', user=user, recommendations=recommendations)

# API endpoint to get recommendation history
@app.route('/api/recommendation-history', methods=['GET'])
def get_recommendation_history():
    if 'user_id' not in session:
        return {'error': 'User not logged in'}, 401
        
    try:
        recommendations = Recommendation.query.filter_by(user_id=session['user_id']).order_by(Recommendation.date.desc()).all()
        result = []
        
        for rec in recommendations:
            result.append({
                'id': rec.id,
                'date': rec.date.strftime('%Y-%m-%d %H:%M:%S'),
                'field_label': rec.field_label or '',
                'n': rec.nitrogen,
                'p': rec.phosphorus,
                'k': rec.potassium,
                'temperature': rec.temperature,
                'humidity': rec.humidity,
                'ph': rec.ph,
                'rainfall': rec.rainfall,
                'crop': rec.recommended_crop,
                'alternatives': rec.alternatives or ''
            })
            
        return result
        
    except Exception as e:
        logging.error(f"Error fetching recommendation history: {e}")
        return {'error': 'Failed to load recommendation history'}, 500

# Sign-Up Form Submit Route
@app.route('/sign_up_submit', methods=['POST'])
def sign_up_submit():
    try:
        username = request.form.get('username', '')
        email = request.form.get('email', '')
        password = request.form.get('password', '')
        if not (username and email and password):
            flash("Please fill in all fields.")
            return redirect(url_for('sign_up'))
            
        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered. Please login.")
            return redirect(url_for('sign_up'))
            
        # Create new user
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        
        # Store user ID in session
        session['user_id'] = new_user.id
        flash("Welcome to CropSense.")
        return redirect(url_for('home'))
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error during signup: {e}")
        flash("An error occurred during sign up.")
        return redirect(url_for('sign_up'))

@app.route('/api/me', methods=['GET'])
def get_current_user():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            return {
                'authenticated': True,
                'username': user.username,
                'avatar': '/static/default-avatar.png'
            }
    return {'authenticated': False}

# Login route
@app.route('/login', methods=['POST'])
def login():
    try:
        email = request.form.get('email', '')
        password = request.form.get('password', '')
        
        user = User.query.filter_by(email=email).first()
        
        if user and user.password == password:  # In a real app, use password hashing!
            session['user_id'] = user.id
            flash("Do small things with love")
            return redirect(url_for('home'))
        else:
            flash("Invalid email or password.")
            return redirect(url_for('sign_up'))
            
    except Exception as e:
        logging.error(f"Error during login: {e}")
        flash("An error occurred during login.")
        return redirect(url_for('sign_up'))

app.secret_key = 'any_random_string'
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("You have been logged out.")
    return redirect(url_for('home'))  # or wherever you want to send them
    
# API endpoint to fetch weather data by city
@app.route('/api/location/<city>', methods=['GET'])
def get_location_data(city):
    location = Location.query.filter_by(city=city).first()
    if location:
        return {
            'temperature': location.temperature,
            'humidity': location.humidity,
            'rainfall': location.rainfall
        }
    return {'error': 'Location not found'}, 404

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = {
            'N': request.form.get('N', ''),
            'P': request.form.get('P', ''),
            'K': request.form.get('K', ''),
            'temperature': request.form.get('temperature', ''),
            'humidity': request.form.get('humidity', ''),
            'ph': request.form.get('ph', ''),
            'rainfall': request.form.get('rainfall', ''),
            'label': request.form.get('label', '')
        }

        logging.info(f"Received form data: {form_data}")

        # Validate input data
        try:
            input_values = [float(form_data[field]) for field in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

            if not (0 <= float(form_data['ph']) <= 14):
                raise ValueError("⚠️ pH must be between 0 and 14")

            if not (0 <= float(form_data['humidity']) <= 100):
                raise ValueError("⚠️ Humidity must be between 0 and 100%")

        except ValueError as ve:
            logging.error(f"Validation error: {ve}")
            return render_template('index.html', error=f"❌ Invalid input: {str(ve)}", show_result=True, form_data=form_data)

        logging.info(f"Validated input data: {input_values}")

        # Preprocess input
        test_input = np.array([input_values])
        test_input_scaled = scaler.transform(test_input)
        test_input_pca = pca.transform(test_input_scaled)

        field_label = form_data['label']
        field_info = f" for {field_label}" if field_label else ""

        # Predict crop
        predicted_label_index = rf_model.predict(test_input_pca)[0]
        predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]

        logging.info(f"Predicted Crop: {predicted_label}")

        # Get top 3 alternative crops
        try:
            proba = rf_model.predict_proba(test_input_pca)[0]
            top_indices = proba.argsort()[-4:-1][::-1]
            alternatives = [label_encoder.inverse_transform([idx])[0] for idx in top_indices]
            alternatives_str = ", ".join(alternatives)
        except:
            alternatives_str = "Not available"
            
        if request.method == 'POST':
            form_data = request.form  # ✅ Step 1
            city = form_data.get('city', '').strip()  # ✅ Step 2 — Extract optional city
        # Save recommendation to database if user is logged in
        if 'user_id' in session:
            try:
                recommendation = Recommendation(
                    user_id=session['user_id'],
                    city=city if city else None,
                    nitrogen=float(form_data['N']),
                    phosphorus=float(form_data['P']),
                    potassium=float(form_data['K']),
                    temperature=float(form_data['temperature']),
                    humidity=float(form_data['humidity']),
                    ph=float(form_data['ph']),
                    rainfall=float(form_data['rainfall']),
                    recommended_crop=predicted_label,
                    alternatives=alternatives_str,
                    field_label=field_label
                )
                db.session.add(recommendation)
                db.session.commit()
                logging.info(f"Recommendation saved for user {session['user_id']}")
            except Exception as e:
                db.session.rollback()
                logging.error(f"Error saving recommendation: {e}")

        return render_template('index.html',
                               prediction=predicted_label,  # Changed to just return the crop name
                               city=city if 'city' in locals() else "",  # Pass city for display
                               alternatives=alternatives_str,
                               show_result=True,
                               form_data=form_data,
                               crop_details=crop_details,  # Pass crop_details to template
                               cities=[location.city for location in Location.query.all()])

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return render_template('index.html', 
                              error=f"⚠️ Error processing request: {str(e)}", 
                              show_result=True, 
                              form_data=request.form,
                              crop_details=crop_details,  # Pass crop_details even in error case
                              cities=[location.city for location in Location.query.all()])
        

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create database tables before running the app
    app.run(debug=True)
    