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
import json
from flask import copy_current_request_context  # Pour garder le contexte



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


@app.route('/generate_advice_stream', methods=['POST', 'GET'])
def generate_advice_stream():
    """Endpoint pour streaming avec Server-Sent Events - Version corrigée"""
    
    try:
        # Parsing des paramètres selon la méthode
        if request.method == 'GET':
            expenses_json = request.args.get('expenses', '[]')
            try:
                expenses = json.loads(expenses_json)
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid JSON in 'expenses' parameter"}), 400
            
            language = request.args.get('language', 'english')
            tone = request.args.get('tone', 'formal')
        else:  # POST
            data = request.get_json(force=True)
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
                
            expenses = data.get('expenses', [])
            language = data.get('language', 'english')
            tone = data.get('tone', 'formal')

        # Validation robuste
        if not expenses or not isinstance(expenses, list):
            return jsonify({"error": "Expenses must be a non-empty list"}), 400
            
        # Nettoyer et valider les dépenses
        clean_expenses = [str(expense).strip() for expense in expenses if str(expense).strip()]
        if not clean_expenses:
            return jsonify({"error": "No valid expenses provided"}), 400

        logger.info(f"Processing {len(clean_expenses)} expenses for streaming")

    except Exception as e:
        logger.error(f"Request parsing failed: {str(e)}")
        return jsonify({"error": f"Request parsing failed: {str(e)}"}), 400

    def generate():
        """Générateur pour le streaming - Version corrigée"""
        try:
            # Initialiser l'advisor
            advisor = get_advisor()
            
            # Log de début
            logger.info("Starting advice generation stream")
            
            # Générer le stream
            chunk_count = 0
            has_content = False
            
            for chunk in advisor.generate_advice_stream(clean_expenses, language, tone):
                chunk_count += 1
                
                if chunk and chunk.strip():
                    has_content = True
                    
                    if request.method == 'GET':
                        # Format SSE pour GET
                        yield f"data: {json.dumps({'content': chunk, 'done': False})}\n\n"
                    else:
                        # Format HTML direct pour POST
                        yield chunk
                        
                # Log périodique
                if chunk_count % 50 == 0:
                    logger.info(f"Streamed {chunk_count} chunks")

            # Finaliser le stream
            if request.method == 'GET':
                yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
            
            # Log final
            logger.info(f"Stream completed with {chunk_count} chunks, has_content: {has_content}")
            
            # Si aucun contenu n'a été généré
            if not has_content:
                logger.warning("No content was generated by the advisor")
                error_content = "<body><h3>No Content</h3><p>No advice content was generated</p></body>"
                if request.method == 'GET':
                    yield f"data: {json.dumps({'content': error_content, 'done': True})}\n\n"
                else:
                    yield error_content

        except Exception as e:
            logger.error(f"Error in stream generation: {str(e)}", exc_info=True)
            error_msg = f"<body><h3>Stream Error</h3><p>Error generating advice: {str(e)}</p></body>"
            
            if request.method == 'GET':
                yield f"data: {json.dumps({'content': error_msg, 'done': True})}\n\n"
            else:
                yield error_msg

    # Headers appropriés
    headers = {
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization'
    }

    # Retourner la réponse selon la méthode
    if request.method == 'GET':
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers=headers
        )
    else:
        return Response(
            generate(),
            mimetype='text/html',
            headers=headers
        )


# Version simplifiée alternative (recommandée)
@app.route('/generate_advice_stream_simple', methods=['POST'])
def generate_advice_stream_simple():
    """Version simplifiée et plus fiable pour streaming POST"""
    try:
        # Validation des données
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        expenses = data.get('expenses', [])
        language = data.get('language', 'english')
        tone = data.get('tone', 'formal')
        
        # Validation
        if not expenses or not isinstance(expenses, list):
            return jsonify({"error": "Expenses must be a non-empty list"}), 400
            
        clean_expenses = [str(exp).strip() for exp in expenses if str(exp).strip()]
        if not clean_expenses:
            return jsonify({"error": "No valid expenses provided"}), 400
        
        logger.info(f"Simple stream: processing {len(clean_expenses)} expenses")
        
        # Générer le stream
        advisor = get_advisor()
        
        def stream_generator():
            try:
                for chunk in advisor.generate_advice_stream(clean_expenses, language, tone):
                    if chunk:
                        yield chunk
            except Exception as e:
                logger.error(f"Stream generation error: {str(e)}")
                yield f"<body><h3>Error</h3><p>Stream error: {str(e)}</p></body>"
        
        return Response(
            stream_generator(),
            mimetype='text/html',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )
        
    except Exception as e:
        logger.error(f"Error in simple stream endpoint: {str(e)}")
        return jsonify({"error": f"Internal error: {str(e)}"}), 500



@app.route('/predict_expense/<income>/<int:bedrooms>/<int:vehicles>/<int:members>/<int:employed>', methods=['GET'])
def predict_expense(income, bedrooms, vehicles, members, employed):
    try:
        # Convert income to float
        try:
            income_float = float(income)
        except ValueError:
            return jsonify({"error": "Income must be a number"}), 400
        
        # Validate other parameters
        if bedrooms < 0 or vehicles < 0 or members < 0 or employed < 0 :
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