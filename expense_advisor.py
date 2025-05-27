# expense_advisor.py
import requests
import json
import time
import gc
import logging
import os
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExpenseAdvisor:
    def __init__(self):
        # Use environment variable for Ollama URL in Azure
        self.ollama_url = os.getenv('OLLAMA_URL', "http://20.169.88.121:11434/api/chat")
        self.model_name = os.getenv('MODEL_NAME', "phi3")
        self.session = None
        logger.info(f"ExpenseAdvisor initialized with URL: {self.ollama_url}")
        
    def get_session(self):
        """Get or create a requests session for connection pooling"""
        if self.session is None:
            self.session = requests.Session()
            # Increased settings for Azure Container Apps with longer timeouts
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=20,
                max_retries=3
            )
            self.session.mount('http://', adapter)
            self.session.mount('https://', adapter)
            
            # Set headers for better compatibility
            self.session.headers.update({
                'User-Agent': 'ExpenseAdvisor/1.0',
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'Connection': 'keep-alive'
            })
        return self.session
    
    @contextmanager
    def cleanup_context(self):
        """Context manager to ensure cleanup after operations"""
        try:
            yield
        finally:
            gc.collect()
    
    def test_connection(self):
        """Test connection to Ollama service"""
        try:
            session = self.get_session()
            # Simple test request with timeout
            test_url = self.ollama_url.replace('/api/chat', '/api/tags')
            response = session.get(test_url, timeout=15)
            logger.info(f"Connection test: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def build_messages(self, expenses, language, tone):
        """Build messages with improved validation and cleanup"""
        try:
            logger.info(f"Building messages with language: {language}, tone: {tone}")
            
            # Validate inputs
            if not expenses or not isinstance(expenses, list):
                raise ValueError("Expenses must be a non-empty list")
            
            # Clean and validate language and tone
            language = str(language).lower().strip()
            tone = str(tone).lower().strip()
            
            # Determine the language instruction
            language_instructions = {
                "english": "Please respond in English.",
                "french": "Veuillez répondre en français."
            }
            language_instruction = language_instructions.get(language, "Please respond in English.")
            
            # Determine the tone instruction
            tone_instructions = {
                "formal": "Use a formal and professional tone in your response.",
                "humorous": "Use a humorous and light-hearted tone in your response.",
                "friendly": "Use a friendly and conversational tone in your response."
            }
            tone_instruction = tone_instructions.get(tone, "Use a formal and professional tone in your response.")
            
            # Build system message
            system_message = (
                f"IMPORTANT: You must respond in {language} language only and with {tone} tone. "
                f"Here is a breakdown of my current expenses. Based on this, can you give me recommendations "
                f"to help me save money. {language_instruction} {tone_instruction}"
            )
            
            # Build user message with better formatting
            expenses_text = "\n".join(f"- {str(expense).strip()}" for expense in expenses if str(expense).strip())
            
            user_message = (
                f"My recent expenses are:\n{expenses_text}\n\n"
                f"IMPORTANT: YOU MUST RESPOND IN {language.upper()} LANGUAGE ONLY. "
                f"{language_instruction} {tone_instruction} "
                f"Format the response in HTML starting from the body tag and give me only recommendations. "
                f"Don't generate anything else in your response. I want the response to be well structured "
                f"because I will visualize it directly on my website. Write the expense first, then the "
                f"recommendation for it, and use the same style for all expenses. Don't use tables."
            )
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            logger.info(f"Messages built successfully for {len(expenses)} expenses")
            return messages
            
        except Exception as e:
            logger.error(f"Error building messages: {str(e)}")
            raise
    
    def generate_advice_stream(self, expenses, language="english", tone="formal"):
        """Generate streaming advice - Version corrigée pour éviter les réponses vides"""
        logger.info(f"Generating streaming advice with language: {language}, tone: {tone}")
        
        # Test de connexion obligatoire
        if not self.test_connection():
            logger.error("Connection test failed before streaming")
            def error_response():
                yield f"<body><h3>Connection Error</h3><p>Unable to connect to AI service at {self.ollama_url}</p></body>"
            return error_response()
        
        with self.cleanup_context():
            try:
                messages = self.build_messages(expenses, language, tone)
                
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 2048,
                        "stop": ["</body>"]  # Arrêter à la fin du body
                    }
                }
                
                session = self.get_session()
                
                def stream_response():
                    inside_think_block = False
                    first_chunk_skipped = False
                    response = None
                    chunk_count = 0
                    accumulated_content = ""
                    last_activity_time = time.time()
                    timeout_threshold = 180  # 3 minutes timeout
                    body_started = False
                    content_generated = False
                    
                    try:
                        logger.info(f"Starting streaming request to {self.ollama_url}")
                        
                        # Requête avec timeouts adaptés
                        response = session.post(
                            self.ollama_url, 
                            json=payload, 
                            stream=True,
                            timeout=(30, 180),  # Timeouts plus longs
                            headers={'Connection': 'keep-alive'}
                        )
                        
                        logger.info(f"Stream response status: {response.status_code}")
                        
                        if response.status_code != 200:
                            error_msg = f"API Error: {response.status_code}"
                            try:
                                error_details = response.text[:500]
                                if error_details:
                                    error_msg += f" - {error_details}"
                            except:
                                pass
                            logger.error(error_msg)
                            yield f"<body><h3>API Error</h3><p>{error_msg}</p></body>"
                            return
                        
                        # Buffer pour accumuler les chunks incomplets
                        buffer = ""
                        
                        for line in response.iter_lines(decode_unicode=True, chunk_size=512):
                            current_time = time.time()
                            
                            # Vérifier le timeout
                            if current_time - last_activity_time > timeout_threshold:
                                logger.warning("Stream timeout detected")
                                if not content_generated:
                                    yield f"<body><h3>Timeout</h3><p>Response generation timed out after {timeout_threshold} seconds</p></body>"
                                else:
                                    yield f"\n<p><em>Response timed out</em></p></body>"
                                break
                            
                            if not line or not line.strip():
                                continue
                            
                            last_activity_time = current_time
                            chunk_count += 1
                            
                            # Log de progression
                            if chunk_count % 10 == 0:
                                logger.debug(f"Processed {chunk_count} chunks")
                                
                            try:
                                # Decoder si nécessaire
                                if isinstance(line, bytes):
                                    line = line.decode('utf-8', errors='ignore')
                                
                                # Ajouter au buffer
                                buffer += line
                                
                                # Essayer de parser le JSON
                                try:
                                    chunk = json.loads(buffer)
                                    buffer = ""  # Reset buffer si parsing réussi
                                except json.JSONDecodeError:
                                    # Si on ne peut pas parser, continuer à accumuler
                                    if len(buffer) > 10000:  # Limite pour éviter les buffers trop grands
                                        logger.warning("Buffer too large, resetting")
                                        buffer = ""
                                    continue
                                
                                # Traiter le chunk
                                content_piece = chunk.get('message', {}).get('content', '')
                                
                                # Vérifier si le stream est terminé
                                if chunk.get('done', False):
                                    logger.info(f"Stream completed by server after {chunk_count} chunks")
                                    if not content_generated:
                                        yield f"<body><h3>No Content</h3><p>AI service completed but generated no content</p></body>"
                                    elif not accumulated_content.rstrip().endswith('</body>'):
                                        yield "</body>"
                                    break
                                
                                if not content_piece:
                                    continue
                                
                                # Filtrer les blocs de réflexion
                                if "<think>" in content_piece:
                                    inside_think_block = True
                                    content_piece = content_piece.split("<think>")[0]
                                if "</think>" in content_piece:
                                    inside_think_block = False
                                    parts = content_piece.split("</think>")
                                    content_piece = parts[-1] if len(parts) > 1 else ""
                                if inside_think_block:
                                    continue
                                
                                # Nettoyer le premier chunk
                                if not first_chunk_skipped:
                                    content_piece = content_piece.lstrip()
                                    if content_piece.startswith('```html'):
                                        content_piece = content_piece.replace('```html', '', 1).lstrip()
                                    elif content_piece.startswith('```'):
                                        content_piece = content_piece.replace('```', '', 1).lstrip()
                                    first_chunk_skipped = True
                                
                                # Nettoyer les marqueurs de fin
                                content_piece = content_piece.replace('```', '')
                                
                                # Accumuler le contenu
                                accumulated_content += content_piece
                                
                                # S'assurer qu'on commence par un body tag
                                if not body_started and content_piece.strip():
                                    if not content_piece.startswith('<body'):
                                        if '<body' not in accumulated_content:
                                            yield '<body>'
                                            body_started = True
                                    else:
                                        body_started = True
                                    content_generated = True
                                
                                # Yielder le contenu si on a du contenu valide
                                if content_piece.strip():
                                    content_generated = True
                                    yield content_piece
                                    
                            except json.JSONDecodeError as e:
                                logger.debug(f"JSON decode error (buffering): {str(e)}")
                                continue
                            except Exception as e:
                                logger.error(f"Error processing stream chunk: {str(e)}")
                                if not content_generated:
                                    yield f"<body><h3>Processing Error</h3><p>Error processing stream: {str(e)}</p></body>"
                                    content_generated = True
                                continue
                        
                        # Finaliser la réponse
                        if content_generated:
                            if not accumulated_content.rstrip().endswith('</body>'):
                                yield "</body>"
                            logger.info(f"Stream completed successfully with {chunk_count} chunks")
                        else:
                            logger.warning("No content was generated during streaming")
                            yield "<body><h3>Empty Response</h3><p>No content was generated by the AI service</p></body>"
                                
                    except requests.exceptions.Timeout as e:
                        error_msg = f"Request timeout after 3 minutes: {str(e)}"
                        logger.error(error_msg)
                        if not content_generated:
                            yield f"<body><h3>Timeout Error</h3><p>{error_msg}</p></body>"
                        else:
                            yield f"\n<p><em>Connection timed out</em></p></body>"
                            
                    except requests.exceptions.ConnectionError as e:
                        error_msg = f"Connection error to {self.ollama_url}: {str(e)}"
                        logger.error(error_msg)
                        if not content_generated:
                            yield f"<body><h3>Connection Error</h3><p>{error_msg}</p></body>"
                        else:
                            yield f"\n<p><em>Connection lost</em></p></body>"
                            
                    except Exception as e:
                        error_msg = f"Unexpected streaming error: {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        if not content_generated:
                            yield f"<body><h3>Streaming Error</h3><p>{error_msg}</p></body>"
                        else:
                            yield f"\n<p><em>Unexpected error occurred</em></p></body>"
                            
                    finally:
                        if response:
                            try:
                                response.close()
                            except:
                                pass
                        gc.collect()
                
                return stream_response()
                
            except Exception as e:
                logger.error(f"Error initializing stream: {str(e)}", exc_info=True)
                def error_response():
                    yield f"<body><h3>Initialization Error</h3><p>Failed to initialize advice stream: {str(e)}</p></body>"
                return error_response()

    def test_connection(self):
        try:
            session = self.get_session()
            
            # Test avec l'endpoint tags d'abord
            test_url = self.ollama_url.replace('/api/chat', '/api/tags')
            logger.info(f"Testing connection to {test_url}")
            
            response = session.get(test_url, timeout=10)
            logger.info(f"Tags endpoint test: {response.status_code}")
            
            if response.status_code == 200:
                return True
            
            # Si tags ne marche pas, tester avec un petit message
            test_payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "test"}],
                "stream": False,
                "options": {"num_predict": 1}
            }
            
            logger.info(f"Testing chat endpoint {self.ollama_url}")
            response = session.post(self.ollama_url, json=test_payload, timeout=15)
            logger.info(f"Chat endpoint test: {response.status_code}")
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def generate_advice(self, expenses, language="english", tone="formal"):
        """Generate non-streaming advice - Azure Container Compatible"""
        logger.info(f"Generating advice with language: {language}, tone: {tone}")
        
        # Test connection first
        if not self.test_connection():
            logger.error("Connection test failed before generating advice")
            return "<body><h3>Connection Error</h3><p>Unable to connect to AI service</p></body>"
        
        with self.cleanup_context():
            try:
                messages = self.build_messages(expenses, language, tone)
                
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 2048
                    }
                }
                
                session = self.get_session()
                
                logger.info("Sending non-streaming request...")
                response = session.post(
                    self.ollama_url, 
                    json=payload,
                    timeout=(30, 120)  # (connect_timeout, read_timeout)
                )
                
                logger.info(f"Non-streaming response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info("API response received successfully")
                    
                    full_content = data.get('message', {}).get('content', '')
                    
                    if not full_content:
                        logger.error("No content received from AI service")
                        return "<body><h3>Error</h3><p>No content received from AI service</p></body>"
                    
                    # Clean markdown fences
                    if full_content.startswith("```html"):
                        full_content = full_content.replace("```html", "").rstrip("```").strip()
                    
                    logger.info(f"Generated advice length: {len(full_content)} characters")
                    return full_content
                else:
                    error_msg = f"API Error: {response.status_code}"
                    if response.text:
                        error_msg += f" - {response.text[:200]}"
                    logger.error(error_msg)
                    return f"<body><h3>Error</h3><p>{error_msg}</p></body>"
                    
            except requests.exceptions.Timeout as e:
                error_msg = f"Request timeout - the AI service took too long to respond: {str(e)}"
                logger.error(error_msg)
                return f"<body><h3>Timeout Error</h3><p>{error_msg}</p></body>"
            except requests.exceptions.ConnectionError as e:
                error_msg = f"Connection error - unable to reach the AI service: {str(e)}"
                logger.error(error_msg)
                return f"<body><h3>Connection Error</h3><p>{error_msg}</p></body>"
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                logger.error(error_msg)
                return f"<body><h3>Error</h3><p>{error_msg}</p></body>"
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'session') and self.session:
            try:
                self.session.close()
            except:
                pass