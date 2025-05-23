from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import logging
import time
import os
from expense_advisor import ExpenseAdvisor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type", "Authorization"]}})

# Configuration
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
MODEL_NAME = os.environ.get('MODEL_NAME', 'phi3')

# Create a single instance of the advisor
advisor = ExpenseAdvisor(ollama_url=OLLAMA_URL, model_name=MODEL_NAME)

@app.route('/', methods=['GET'])
def home():
    """Endpoint d'accueil"""
    return jsonify({
        "message": "Expense Advisor API is running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "generate_advice": "/generate_advice (POST)",
            "models": "/models"
        },
        "timestamp": time.time()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de vérification de santé"""
    ollama_status = advisor.check_ollama_status()
    available_models = advisor.get_available_models() if ollama_status else []
    
    return jsonify({
        "status": "healthy" if ollama_status else "unhealthy",
        "ollama_available": ollama_status,
        "ollama_url": OLLAMA_URL,
        "current_model": MODEL_NAME,
        "available_models": available_models,
        "timestamp": time.time(),
        "port": "5000"
    })

@app.route('/models', methods=['GET'])
def get_models():
    """Endpoint pour récupérer les modèles disponibles"""
    try:
        if not advisor.check_ollama_status():
            return jsonify({"error": "Ollama service is not available"}), 503
        
        available_models = advisor.get_available_models()
        if available_models is None:
            return jsonify({"error": "Failed to retrieve models"}), 500
        
        return jsonify({
            "current_model": MODEL_NAME,
            "available_models": available_models,
            "total_models": len(available_models)
        })
        
    except Exception as e:
        logger.error(f"Error in get_models: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/generate_advice', methods=['POST'])
def generate_advice():
    """Endpoint principal pour générer des conseils de dépenses"""
    try:
        logger.info("Generate advice endpoint called")
        data = request.get_json()
        
        if not data:
            logger.error("No JSON data received")
            return jsonify({"error": "No data provided"}), 400
        
        # Validation des données d'entrée
        expenses = data.get('expenses', [])
        language = data.get('language', "english").lower()
        tone = data.get('tone', "formal").lower()
        
        # Validation des paramètres
        valid_languages = ["english", "french"]
        valid_tones = ["formal", "humorous", "friendly"]
        
        if language not in valid_languages:
            return jsonify({
                "error": f"Invalid language. Must be one of: {', '.join(valid_languages)}"
            }), 400
        
        if tone not in valid_tones:
            return jsonify({
                "error": f"Invalid tone. Must be one of: {', '.join(valid_tones)}"
            }), 400
        
        logger.info(f"Request received with language: {language}, tone: {tone}")
        logger.info(f"Expenses count: {len(expenses)}")
        
        if not expenses or not isinstance(expenses, list):
            logger.error("Invalid expenses data")
            return jsonify({"error": "Please provide a list of expenses."}), 400
        
        if len(expenses) == 0:
            return jsonify({"error": "Expenses list cannot be empty."}), 400
        
        if len(expenses) > 50:  # Limite raisonnable
            return jsonify({"error": "Too many expenses. Maximum 50 expenses allowed."}), 400
        
        # Convertir les dépenses en chaînes et nettoyer
        expenses = [str(expense).strip() for expense in expenses if str(expense).strip()]
        
        if not expenses:
            return jsonify({"error": "No valid expenses provided after cleaning."}), 400
        
        logger.info(f"Processing {len(expenses)} expenses with language={language}, tone={tone}")
        
        # Vérifier qu'Ollama est disponible avant de traiter
        if not advisor.check_ollama_status():
            logger.error("Ollama service unavailable")
            return jsonify({
                "error": "AI service is currently unavailable. Please try again later.",
                "suggestion": "Make sure Ollama is running and the model is loaded."
            }), 503
        
        # Retourner la réponse en streaming
        return Response(
            advisor.generate_advice_stream(expenses, language, tone),
            mimetype='text/plain',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in generate_advice: {e}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e) if app.debug else "Please try again later"
        }), 500

@app.route('/generate_advice_sync', methods=['POST']) 
def generate_advice_sync():
    """Version synchrone de l'endpoint de génération de conseils"""
    try:
        logger.info("Generate advice sync endpoint called")
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        expenses = data.get('expenses', [])
        language = data.get('language', "english").lower()
        tone = data.get('tone', "formal").lower()
        
        # Mêmes validations que l'endpoint streaming
        valid_languages = ["english", "french"]
        valid_tones = ["formal", "humorous", "friendly"]
        
        if language not in valid_languages:
            return jsonify({"error": f"Invalid language. Must be one of: {', '.join(valid_languages)}"}), 400
        
        if tone not in valid_tones:
            return jsonify({"error": f"Invalid tone. Must be one of: {', '.join(valid_tones)}"}), 400
        
        if not expenses or not isinstance(expenses, list) or len(expenses) == 0:
            return jsonify({"error": "Please provide a non-empty list of expenses."}), 400
        
        expenses = [str(expense).strip() for expense in expenses if str(expense).strip()]
        
        if not advisor.check_ollama_status():
            return jsonify({"error": "AI service is currently unavailable."}), 503
        
        # Générer la réponse complète
        advice = advisor.generate_advice(expenses, language, tone)
        
        return jsonify({
            "advice": advice,
            "language": language,
            "tone": tone,
            "expenses_count": len(expenses),
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"Error in generate_advice_sync: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def not_found(error):
    """Gestionnaire d'erreur 404"""
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/health", "/models", "/generate_advice", "/generate_advice_sync"]
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Gestionnaire d'erreur 405"""
    return jsonify({
        "error": "Method not allowed",
        "suggestion": "Check the HTTP method and endpoint documentation"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """Gestionnaire d'erreur 500"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "error": "Internal server error",
        "suggestion": "Please try again later"
    }), 500

if __name__ == '__main__':
    logger.info("Starting Flask app on port 5000...")
    
    # Attendre qu'Ollama soit prêt au démarrage direct
    if not advisor.wait_for_ollama():
        logger.error("Ollama failed to start within timeout period")
        logger.info("Starting Flask app anyway - Ollama can be started later")
    
    app.run(host='0.0.0.0', port=5000, debug=False)