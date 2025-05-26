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
        """Generate streaming advice - Fixed for Azure Container Apps with better error handling"""
        logger.info(f"Generating streaming advice with language: {language}, tone: {tone}")
        
        # Test connection first
        if not self.test_connection():
            logger.error("Connection test failed before streaming")
            def error_response():
                yield f"<body><h3>Connection Error</h3><p>Unable to connect to AI service</p></body>"
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
                        "num_predict": 2048  # Limit response length to prevent timeouts
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
                    timeout_threshold = 120  # 2 minutes timeout
                    
                    try:
                        logger.info("Starting streaming request...")
                        
                        # Use longer timeout and chunk size for Azure
                        response = session.post(
                            self.ollama_url, 
                            json=payload, 
                            stream=True,
                            timeout=(30, 120),  # (connect_timeout, read_timeout)
                            headers={'Connection': 'keep-alive'}
                        )
                        
                        logger.info(f"Stream response status: {response.status_code}")
                        
                        if response.status_code != 200:
                            error_msg = f"API Error: {response.status_code}"
                            if response.text:
                                error_msg += f" - {response.text[:200]}"
                            logger.error(error_msg)
                            yield f"<body><h3>Error</h3><p>{error_msg}</p></body>"
                            return
                        
                        # Start with opening body tag if not present
                        yield_started = False
                        
                        for line in response.iter_lines(decode_unicode=True, chunk_size=1024):
                            current_time = time.time()
                            
                            # Check for timeout
                            if current_time - last_activity_time > timeout_threshold:
                                logger.warning("Stream timeout detected")
                                if not yield_started:
                                    yield f"<body><h3>Timeout</h3><p>Response generation timed out</p></body>"
                                else:
                                    yield f"\n<p><em>Response generation timed out</em></p></body>"
                                break
                            
                            if not line:
                                continue
                            
                            last_activity_time = current_time
                            chunk_count += 1
                            
                            # Log progress every 20 chunks
                            if chunk_count % 20 == 0:
                                logger.info(f"Processed {chunk_count} chunks")
                                
                            try:
                                if isinstance(line, bytes):
                                    line = line.decode('utf-8')
                                    
                                chunk = json.loads(line)
                                content_piece = chunk.get('message', {}).get('content', '')
                                
                                # Check if stream is done
                                if chunk.get('done', False):
                                    logger.info("Stream marked as done by server")
                                    if not yield_started:
                                        yield f"<body><h3>Error</h3><p>No content received</p></body>"
                                    else:
                                        # Ensure proper closing
                                        if not accumulated_content.rstrip().endswith('</body>'):
                                            yield "</body>"
                                    break
                                
                                if not content_piece:
                                    continue
                                
                                # Handle think blocks
                                if "<think>" in content_piece:
                                    inside_think_block = True
                                    continue
                                if "</think>" in content_piece:
                                    inside_think_block = False
                                    continue
                                if inside_think_block:
                                    continue
                                
                                # Clean first chunk
                                if not first_chunk_skipped:
                                    content_piece = content_piece.lstrip()
                                    if content_piece.startswith('```html'):
                                        content_piece = content_piece.replace('```html', '', 1).lstrip()
                                    first_chunk_skipped = True
                                
                                # Clean ending markdown
                                if content_piece.endswith('```'):
                                    content_piece = content_piece.rstrip('```').rstrip()
                                
                                # Accumulate content for tracking
                                accumulated_content += content_piece
                                
                                # Ensure we start with body tag
                                if not yield_started and content_piece.strip():
                                    if not content_piece.startswith('<body'):
                                        # If content doesn't start with body, add it
                                        if '<body' not in accumulated_content:
                                            yield '<body>'
                                    yield_started = True
                                
                                yield content_piece
                                
                            except json.JSONDecodeError as e:
                                logger.warning(f"JSON decode error in stream: {str(e)}")
                                continue
                            except Exception as e:
                                logger.error(f"Error processing stream chunk: {str(e)}")
                                if not yield_started:
                                    yield f"<body><h3>Error</h3><p>Streaming Error: {str(e)}</p></body>"
                                    yield_started = True
                                else:
                                    yield f"\n<p><em>Streaming Error: {str(e)}</em></p>"
                        
                        # Ensure proper closure
                        if yield_started and not accumulated_content.rstrip().endswith('</body>'):
                            yield "</body>"
                        elif not yield_started:
                            yield "<body><h3>No Content</h3><p>No content was received from the AI service</p></body>"
                        
                        logger.info(f"Streaming completed. Total chunks: {chunk_count}")
                                
                    except requests.exceptions.Timeout as e:
                        error_msg = f"Request timeout - the AI service took too long to respond: {str(e)}"
                        logger.error(error_msg)
                        if not yield_started:
                            yield f"<body><h3>Timeout Error</h3><p>{error_msg}</p></body>"
                        else:
                            yield f"\n<p><em>Connection timed out</em></p></body>"
                    except requests.exceptions.ConnectionError as e:
                        error_msg = f"Connection error - unable to reach the AI service: {str(e)}"
                        logger.error(error_msg)
                        if not yield_started:
                            yield f"<body><h3>Connection Error</h3><p>{error_msg}</p></body>"
                        else:
                            yield f"\n<p><em>Connection lost</em></p></body>"
                    except Exception as e:
                        error_msg = f"Unexpected error: {str(e)}"
                        logger.error(error_msg)
                        if not yield_started:
                            yield f"<body><h3>Error</h3><p>{error_msg}</p></body>"
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
                logger.error(f"Error in generate_advice_stream: {str(e)}")
                def error_response():
                    yield f"<body><h3>Error</h3><p>Failed to generate advice: {str(e)}</p></body>"
                return error_response()
    
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