# app.py
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from expense_advisor import ExpenseAdvisor  # Make sure this import works correctly

# Initialize Flask app
device = 0 if torch.cuda.is_available() else -1  # Utiliser 0 pour le GPU, -1 pour CPU
print(f"CUDA available: {torch.cuda.is_available()}")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type", "Authorization"]}})

# Create a single instance of the advisor
advisor = ExpenseAdvisor()

@app.route('/generate_advice', methods=['POST'])
def generate_advice():
    data = request.get_json()
    expenses = data.get('expenses', [])
    
    # Ensure language and tone have default values if not provided
    language = data.get('language')
    tone = data.get('tone')
    
    # Set defaults if None
    if language is None:
        language = "english"
    if tone is None:
        tone = "formal"
    
    # Add more debug information
    print(f"[Route] Request received with language: {language}, tone: {tone}")
    print(f"[Route] Request data: {data}")
    
    if not expenses or not isinstance(expenses, list):
        return jsonify({"error": "Please provide a list of expenses."}), 400
    
    # Convert expenses to strings if they aren't already
    expenses = [str(expense) for expense in expenses]
    
    print(f"[Route] Processing {len(expenses)} expenses with language={language}, tone={tone}")
    
    # Send the stream response
    return Response(advisor.generate_advice_stream(expenses, language, tone), 
                   mimetype='text/plain')



