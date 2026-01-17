from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import os
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load the trained model
# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'attrition_model.pkl')
model = None
cat_features = None
feature_order = None

def load_model():
    """Load the trained CatBoost model from pickle file"""
    global model, cat_features, feature_order
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
                model = model_data['model']
                cat_features = model_data['cat_features']
                feature_order = model_data['feature_order']
            print(f"✓ Model loaded successfully from {MODEL_PATH}")
            print(f"✓ Categorical features: {cat_features}")
            print(f"✓ Feature order: {feature_order}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    else:
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train and save the model first.")

# Load model on startup
try:
    load_model()
except FileNotFoundError as e:
    print(f"Warning: {e}")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict employee attrition based on input features"""
    if model is None or feature_order is None:
        print("ERROR: Model not loaded")
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
    
    try:
        # Get JSON data from request
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided in request'}), 400
        
        print(f"Received prediction request with data: {data}")
        
        # Validate required fields
        required_fields = feature_order
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            error_msg = f'Missing required fields: {", ".join(missing_fields)}'
            print(f"ERROR: {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        # Create DataFrame with single row
        input_data = pd.DataFrame([{
            'Department': str(data['Department']),
            'Overtime': str(data['Overtime']).replace('–', '-'),  # Normalize en-dash to hyphen
            'Promotion_Gap': float(data['Promotion_Gap']),
            'Job_Satisfaction': str(data['Job_Satisfaction']),
            'AI_Automation_Risk': str(data['AI_Automation_Risk']),
            'Recent_Layoffs': str(data['Recent_Layoffs']),
            'Job_Security': str(data['Job_Security']),
            'Market_Demand': str(data['Market_Demand'])
        }])
        
        # Ensure columns are in the correct order expected by the model
        input_data = input_data[feature_order]
        
        print(f"Input data prepared: {input_data.to_dict('records')[0]}")
        print(f"Feature order: {feature_order}")
        print(f"Input columns: {list(input_data.columns)}")
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        print(f"Prediction: {prediction}, Probabilities: {probability}")
        
        # Get feature importance (if available)
        feature_importance = None
        try:
            importance = model.get_feature_importance()
            feature_importance = dict(zip(feature_order, importance.tolist()))
        except Exception as e:
            print(f"Warning: Could not get feature importance: {e}")
        
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Yes (Likely to Leave)' if prediction == 1 else 'No (Likely to Stay)',
            'probability': {
                'stay': float(probability[0]),
                'leave': float(probability[1])
            },
            'confidence': float(max(probability)),
            'feature_importance': feature_importance
        }
        
        print(f"Returning result: {result}")
        return jsonify(result)
    
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR in predict endpoint: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get information about required features and their possible values"""
    return jsonify({
        'features': {
            'Department': {
                'type': 'categorical',
                'description': 'Primary Department'
            },
            'Overtime': {
                'type': 'categorical',
                'description': 'Average Monthly Overtime',
                'options': ['0 hours', '1-10 hours', '11-20 hours', '20+ hours']
            },
            'Promotion_Gap': {
                'type': 'numeric',
                'description': 'Years since last job title change or promotion',
                'min': 0,
                'max': 50
            },
            'Job_Satisfaction': {
                'type': 'categorical',
                'description': 'Job Satisfaction Level',
                'options': ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
            },
            'AI_Automation_Risk': {
                'type': 'categorical',
                'description': 'Risk of AI/Automation',
                'options': ['Very Low', 'Low', 'Medium', 'High', 'Very High']
            },
            'Recent_Layoffs': {
                'type': 'categorical',
                'description': 'Has department experienced layoffs in last 12 months?',
                'options': ['Yes', 'No']
            },
            'Job_Security': {
                'type': 'categorical',
                'description': 'Job Security Level',
                'options': ['Very Unstable', 'Unstable', 'Medium', 'Secure', 'Very Secure']
            },
            'Market_Demand': {
                'type': 'categorical',
                'description': 'Ease of finding similar role elsewhere',
                'options': ['Very Easy', 'Easy', 'Neutral', 'Difficult']
            }
        }
    })

if __name__ == '__main__':
    print(f"\n{'='*50}")
    print("Starting Flask Backend Server")
    print(f"{'='*50}")
    print(f"Server will run on: http://localhost:5001")
    print(f"API endpoint: http://localhost:5001/api/predict")
    print(f"Health check: http://localhost:5001/api/health")
    print(f"{'='*50}\n")
    app.run(debug=True, port=5001, host='0.0.0.0')

