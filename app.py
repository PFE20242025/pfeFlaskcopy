# app.py
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import logging
import time
import requests
import subprocess
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type", "Authorization"]}})

def check_ollama_status():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Ollama check failed: {e}")
        return False

def wait_for_ollama(max_wait=60):
    """Wait for Ollama to be ready"""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if check_ollama_status():
            logger.info("Ollama is ready!")
            return True
        logger.info("Waiting for Ollama to start...")
        time.sleep(2)
    return False

class ExpenseAdvisor:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/chat"
        self.model_name = "phi3"
    
    def build_messages(self, expenses, language, tone):
        logger.debug(f"Building messages with language: {language}, tone: {tone}")
        
        # Determine the language instruction based on user selection
        language_instruction = ""
        if language == "english":
            language_instruction = "Please respond in English."
        elif language == "french":
            language_instruction = "Veuillez répondre en français."
        
        # Determine the tone instruction based on user selection
        tone_instruction = ""
        if tone == "formal":
            tone_instruction = "Use a formal and professional tone in your response."
        elif tone == "humorous":
            tone_instruction = "Use a humorous and light-hearted tone in your response."
        elif tone == "friendly":
            tone_instruction = "Use a friendly and conversational tone in your response."
        
        system_message = f"IMPORTANT: You must respond in {language} language only and with {tone} tone. Here is a breakdown of my current expenses. Based on this, can you give me recommendations to help me save money. {language_instruction} {tone_instruction}"
        
        user_message = (
            f"My recent expenses are:\n" + "\n".join(f"- {expense}" for expense in expenses) +
            f"\n\nIMPORTANT: YOU MUST RESPOND IN {language.upper()} LANGUAGE ONLY. {language_instruction} {tone_instruction} Format the response in HTML starting from the body tag and give me only recommendations. Don't generate anything else in your response. I want the response to be well structured because I will visualize it directly on my website. Write the expense first, then the recommendation for it, and use the same style for all expenses. Don't use tables."
        )
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        logger.debug(f"System message: {system_message}")
        logger.debug(f"First 150 chars of user message: {user_message[:150]}...")
        
        return messages
    
    def generate_advice_stream(self, expenses, language="english", tone="formal"):
        logger.info(f"[ExpenseAdvisor] Generating advice with language: {language}, tone: {tone}")
        
        # Check if Ollama is accessible
        if not check_ollama_status():
            logger.error("Ollama is not accessible!")
            yield "Error: Ollama service is not available. Please check if the service is running."
            return
        
        messages = self.build_messages(expenses, language, tone)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True
        }

        def stream_response():
            inside_think_block = False
            first_chunk_skipped = False
            response_started = False

            try:
                logger.info(f"Making request to Ollama: {self.ollama_url}")
                with requests.post(self.ollama_url, json=payload, stream=True, timeout=30) as response:
                    logger.info(f"Ollama response status: {response.status_code}")
                    
                    if response.status_code == 200:
                        for line in response.iter_lines(decode_unicode=True):
                            if line:
                                try:
                                    if isinstance(line, bytes):
                                        line = line.decode('utf-8')
                                    
                                    chunk = json.loads(line)
                                    content_piece = chunk.get('message', {}).get('content', '')
                                    
                                    if content_piece:
                                        response_started = True
                                        logger.debug(f"Content piece: {content_piece[:50]}...")
                                        
                                        # Skip thinking blocks
                                        if "<think>" in content_piece:
                                            inside_think_block = True
                                            continue
                                        if "</think>" in content_piece:
                                            inside_think_block = False
                                            continue
                                        if inside_think_block:
                                            continue

                                        # Remove the first "```html" manually
                                        if not first_chunk_skipped:
                                            content_piece = content_piece.lstrip()
                                            if content_piece.startswith('```html'):
                                                content_piece = content_piece.replace('```html', '', 1).lstrip()
                                            first_chunk_skipped = True

                                        yield content_piece
                                        
                                    # Check if response is done
                                    if chunk.get('done', False):
                                        logger.info("Ollama response completed")
                                        break
                                        
                                except json.JSONDecodeError as e:
                                    logger.error(f"JSON decode error: {e}, line: {line}")
                                    continue
                                except Exception as e:
                                    logger.error(f"Streaming error: {e}")
                                    yield f"\n[Streaming Error]: {str(e)}\n"
                        
                        if not response_started:
                            logger.warning("No content received from Ollama")
                            yield "No response generated. The model might be loading or unavailable."
                            
                    else:
                        error_msg = f"Ollama API Error: {response.status_code} - {response.text}"
                        logger.error(error_msg)
                        yield error_msg
                        
            except requests.exceptions.Timeout:
                error_msg = "Request to Ollama timed out"
                logger.error(error_msg)
                yield error_msg
            except requests.exceptions.ConnectionError:
                error_msg = "Cannot connect to Ollama service"
                logger.error(error_msg)
                yield error_msg
            except Exception as e:
                error_msg = f"Connection Error: {str(e)}"
                logger.error(error_msg)
                yield error_msg

        return stream_response()

# Create a single instance of the advisor
advisor = ExpenseAdvisor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    ollama_status = check_ollama_status()
    return jsonify({
        "status": "healthy" if ollama_status else "unhealthy",
        "ollama_available": ollama_status,
        "timestamp": time.time()
    })

@app.route('/generate_advice', methods=['POST'])
def generate_advice():
    try:
        logger.info("Generate advice endpoint called")
        data = request.get_json()
        
        if not data:
            logger.error("No JSON data received")
            return jsonify({"error": "No data provided"}), 400
        
        expenses = data.get('expenses', [])
        language = data.get('language', "english")
        tone = data.get('tone', "formal")
        
        logger.info(f"[Route] Request received with language: {language}, tone: {tone}")
        logger.info(f"[Route] Expenses count: {len(expenses)}")
        
        if not expenses or not isinstance(expenses, list):
            logger.error("Invalid expenses data")
            return jsonify({"error": "Please provide a list of expenses."}), 400
        
        # Convert expenses to strings if they aren't already
        expenses = [str(expense) for expense in expenses]
        
        logger.info(f"[Route] Processing {len(expenses)} expenses with language={language}, tone={tone}")
        
        # Check if Ollama is available before processing
        if not check_ollama_status():
            logger.error("Ollama service unavailable")
            return jsonify({"error": "AI service is currently unavailable. Please try again later."}), 503
        
        # Return the streaming response
        return Response(
            advisor.generate_advice_stream(expenses, language, tone),
            mimetype='text/plain',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in generate_advice: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Wait for Ollama to be ready when running directly
    logger.info("Starting Flask app...")
    if not wait_for_ollama():
        logger.error("Ollama failed to start within timeout period")
    app.run(host='0.0.0.0', port=5000, debug=True)