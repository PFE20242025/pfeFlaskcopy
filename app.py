# app.py

from flask import Flask, request, jsonify, Response
from transformers import pipeline
from flask_cors import CORS
from expense_advisor import ExpenseAdvisor
import torch
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import gc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
device = 0 if torch.cuda.is_available() else -1
logger.info(f"CUDA available: {torch.cuda.is_available()}")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type", "Authorization"]}})

# Global variables for models (lazy loading)
classifier = None
advisor = None
expense_model = None
scaler = None

def get_classifier():
    """Lazy load the classifier to save memory"""
    global classifier
    if classifier is None:
        try:
            classifier = pipeline("zero-shot-classification", 
                                model="joeddav/xlm-roberta-large-xnli", 
                                tokenizer="xlm-roberta-large",
                                device=device)
            logger.info("Classifier loaded successfully")
        except Exception as e:
            logger.error(f"Error loading classifier: {str(e)}")
            raise
    return classifier

def get_advisor():
    """Lazy load the advisor"""
    global advisor
    if advisor is None:
        try:
            advisor = ExpenseAdvisor()
            logger.info("Expense advisor loaded successfully")
        except Exception as e:
            logger.error(f"Error loading advisor: {str(e)}")
            raise
    return advisor

def get_expense_model():
    """Lazy load the expense prediction model"""
    global expense_model, scaler
    if expense_model is None or scaler is None:
        try:
            expense_model = pickle.load(open("model_file.pkl", "rb"))
            
            # Initialize scaler
            scaler = StandardScaler()
            scaler.mean_ = np.array([3.71568945e+03, 1.78916516e+00, 8.13038061e-02, 4.63287268e+00, 1.26843566e+00])
            scaler.scale_ = np.array([4.26888540e+03, 1.10490389e+00, 3.46252216e-01, 2.27974326e+00, 1.14820562e+00])
            scaler.var_ = np.array([1.82233825e+07, 1.22081260e+00, 1.19890597e-01, 5.19722931e+00, 1.31837614e+00])
            
            logger.info("Expense model and scaler loaded successfully")
        except Exception as e:
            logger.error(f"Error loading expense model: {str(e)}")
            raise
    return expense_model, scaler

# Categories for classification
categories = ["Food", "Transport", "Entertainment", "Health", "Electronics", "Fashion", "Housing", "Others"]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the product name from the request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        product = data.get('product', '').strip()
        
        if not product:
            return jsonify({'error': 'Product name is required'}), 400
        
        # Get classifier and predict
        classifier = get_classifier()
        result = classifier(product, categories)
        predicted_category = result["labels"][0]
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        return jsonify({
            'product': product,
            'predicted_category': predicted_category
        })
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/generate_advice', methods=['POST'])
def generate_advice():
    """Generate complete advice response (non-streaming)"""
    try:
        logger.info("Received request for generate_advice")
        
        # Validate JSON data
        data = request.get_json()
        if not data:
            logger.error("No JSON data provided")
            return jsonify({"error": "No JSON data provided."}), 400
        
        # Extract parameters
        expenses = data.get('expenses', [])
        language = data.get('language', "english")
        tone = data.get('tone', "formal")
        
        logger.info(f"Request params - Language: {language}, Tone: {tone}, Expenses count: {len(expenses) if expenses else 0}")
        
        # Validate expenses
        if not expenses or not isinstance(expenses, list):
            return jsonify({"error": "Please provide a list of expenses."}), 400
        
        # Clean and validate expenses
        expenses = [str(expense).strip() for expense in expenses if str(expense).strip()]
        if not expenses:
            return jsonify({"error": "Please provide valid expenses."}), 400
        
        logger.info(f"Processing {len(expenses)} valid expenses")
        
        # Generate advice
        advisor = get_advisor()
        advice_html = advisor.generate_advice(expenses, language, tone)
        
        logger.info("Successfully generated advice")
        return Response(advice_html, mimetype='text/html')
        
    except Exception as e:
        logger.error(f"Error in generate_advice endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/generate_advice_stream', methods=['POST'])
def generate_advice_stream():
    """Generate streaming advice response"""
    try:
        logger.info("Received request for generate_advice_stream")
        
        # Validate JSON data
        data = request.get_json()
        if not data:
            logger.error("No JSON data provided")
            return jsonify({"error": "No JSON data provided."}), 400
        
        # Extract parameters
        expenses = data.get('expenses', [])
        language = data.get('language', "english")
        tone = data.get('tone', "formal")
        
        logger.info(f"Streaming request params - Language: {language}, Tone: {tone}, Expenses count: {len(expenses) if expenses else 0}")
        
        # Validate expenses
        if not expenses or not isinstance(expenses, list):
            return jsonify({"error": "Please provide a list of expenses."}), 400
        
        # Clean and validate expenses
        expenses = [str(expense).strip() for expense in expenses if str(expense).strip()]
        if not expenses:
            return jsonify({"error": "Please provide valid expenses."}), 400
        
        logger.info(f"Processing {len(expenses)} valid expenses for streaming")
        
        # Generate streaming advice
        advisor = get_advisor()
        
        def cleanup_generator():
            try:
                chunk_count = 0
                for chunk in advisor.generate_advice_stream(expenses, language, tone):
                    chunk_count += 1
                    yield chunk
                logger.info(f"Streaming completed with {chunk_count} chunks")
            except Exception as e:
                logger.error(f"Error in streaming generator: {str(e)}")
                yield f"<body><h3>Streaming Error</h3><p>{str(e)}</p></body>"
            finally:
                gc.collect()
        
        return Response(cleanup_generator(), mimetype='text/html')
        
    except Exception as e:
        logger.error(f"Error in generate_advice_stream endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
    



@app.route('/predict_expense/<income>/<int:bedrooms>/<int:vehicles>/<int:members>/<int:employed>', methods=['GET'])
def predict_expense(income, bedrooms, vehicles, members, employed):
    try:
        # Convert income to float
        try:
            income_float = float(income)
        except ValueError:
            return jsonify({"error": "Income must be a number"}), 400
        
        # Validate other parameters
        if bedrooms < 0 or vehicles < 0 or members < 0 or employed not in [0, 1]:
            return jsonify({"error": "Invalid parameter values"}), 400
            
        # Get model and scaler
        model, scaler = get_expense_model()
        
        # Prepare input
        input_data = np.array([[income_float, bedrooms, vehicles, members, employed]])
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)
        
        # Force garbage collection
        gc.collect()
        
        response = {
            'expense_prediction': prediction.tolist()
        }
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in predict_expense endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)