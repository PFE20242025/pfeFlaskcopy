import requests
import json
import logging
import time

# Set up logging
logger = logging.getLogger(__name__)

class ExpenseAdvisor:
    """
    Classe pour gérer les conseils de dépenses via Ollama
    """
    
    def __init__(self, ollama_url="http://localhost:11434", model_name="phi3"):
        """
        Initialise l'ExpenseAdvisor
        
        Args:
            ollama_url (str): URL de base d'Ollama
            model_name (str): Nom du modèle à utiliser
        """
        self.ollama_url = f"{ollama_url}/api/chat"
        self.model_name = model_name
        self.tags_url = f"{ollama_url}/api/tags"
        
    def check_ollama_status(self):
        """
        Vérifie si Ollama est en marche et accessible
        
        Returns:
            bool: True si Ollama est accessible, False sinon
        """
        try:
            response = requests.get(self.tags_url, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama check failed: {e}")
            return False
    
    def wait_for_ollama(self, max_wait=120):
        """
        Attend qu'Ollama soit prêt
        
        Args:
            max_wait (int): Temps maximum d'attente en secondes
            
        Returns:
            bool: True si Ollama est prêt, False si timeout
        """
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if self.check_ollama_status():
                logger.info("Ollama is ready!")
                return True
            logger.info("Waiting for Ollama to start...")
            time.sleep(5)
        logger.error("Ollama failed to start within timeout period")
        return False
    
    def get_available_models(self):
        """
        Récupère la liste des modèles disponibles
        
        Returns:
            list: Liste des modèles disponibles ou None si erreur
        """
        try:
            response = requests.get(self.tags_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return None
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return None
    
    def build_system_message(self, language, tone):
        """
        Construit le message système basé sur la langue et le ton
        
        Args:
            language (str): Langue souhaitée ('english', 'french')
            tone (str): Ton souhaité ('formal', 'humorous', 'friendly')
            
        Returns:
            str: Message système formaté
        """
        # Instructions de langue
        language_instructions = {
            "english": "Please respond in English only.",
            "french": "Veuillez répondre en français uniquement."
        }
        
        # Instructions de ton
        tone_instructions = {
            "formal": "Use a formal and professional tone in your response.",
            "humorous": "Use a humorous and light-hearted tone in your response.",
            "friendly": "Use a friendly and conversational tone in your response."
        }
        
        language_instruction = language_instructions.get(language, language_instructions["english"])
        tone_instruction = tone_instructions.get(tone, tone_instructions["formal"])
        
        system_message = (
            f"You are a financial advisor helping users optimize their expenses. "
            f"IMPORTANT: You must respond in {language} language only and with {tone} tone. "
            f"{language_instruction} {tone_instruction} "
            f"Provide practical and actionable recommendations to help save money."
        )
        
        return system_message
    
    def build_user_message(self, expenses, language, tone):
        """
        Construit le message utilisateur avec les dépenses
        
        Args:
            expenses (list): Liste des dépenses
            language (str): Langue souhaitée
            tone (str): Ton souhaité
            
        Returns:
            str: Message utilisateur formaté
        """
        language_instructions = {
            "english": "Please respond in English only.",
            "french": "Veuillez répondre en français uniquement."
        }
        
        tone_instructions = {
            "formal": "Use a formal and professional tone in your response.",
            "humorous": "Use a humorous and light-hearted tone in your response.",
            "friendly": "Use a friendly and conversational tone in your response."
        }
        
        language_instruction = language_instructions.get(language, language_instructions["english"])
        tone_instruction = tone_instructions.get(tone, tone_instructions["formal"])
        
        expenses_text = "\n".join(f"- {expense}" for expense in expenses)
        
        user_message = (
            f"Here are my recent expenses:\n{expenses_text}\n\n"
            f"IMPORTANT: YOU MUST RESPOND IN {language.upper()} LANGUAGE ONLY. "
            f"{language_instruction} {tone_instruction}\n\n"
            f"Please provide specific recommendations for each expense to help me save money. "
            f"Format the response in clean HTML starting from the body tag. "
            f"Structure it well as it will be displayed on a website. "
            f"For each expense, write the expense name first, then your recommendation. "
            f"Use consistent styling throughout. Do not use tables."
        )
        
        return user_message
    
    def build_messages(self, expenses, language="english", tone="formal"):
        """
        Construit la liste complète des messages pour Ollama
        
        Args:
            expenses (list): Liste des dépenses
            language (str): Langue souhaitée
            tone (str): Ton souhaité
            
        Returns:
            list: Liste des messages formatés pour l'API Ollama
        """
        logger.info(f"Building messages with language: {language}, tone: {tone}")
        
        system_message = self.build_system_message(language, tone)
        user_message = self.build_user_message(expenses, language, tone)
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        logger.debug(f"System message: {system_message[:100]}...")
        logger.debug(f"User message: {user_message[:150]}...")
        
        return messages
    
    def clean_response_content(self, content_piece, inside_think_block, first_chunk_skipped):
        """
        Nettoie le contenu de la réponse
        
        Args:
            content_piece (str): Morceau de contenu à nettoyer
            inside_think_block (bool): Si on est dans un bloc de réflexion
            first_chunk_skipped (bool): Si le premier chunk a été traité
            
        Returns:
            tuple: (contenu nettoyé, inside_think_block, first_chunk_skipped)
        """
        # Skip thinking blocks
        if "<think>" in content_piece:
            inside_think_block = True
            return "", inside_think_block, first_chunk_skipped
        if "</think>" in content_piece:
            inside_think_block = False
            return "", inside_think_block, first_chunk_skipped
        if inside_think_block:
            return "", inside_think_block, first_chunk_skipped

        # Remove markdown code blocks
        if not first_chunk_skipped:
            content_piece = content_piece.lstrip()
            if content_piece.startswith('```html'):
                content_piece = content_piece.replace('```html', '', 1).lstrip()
            elif content_piece.startswith('```'):
                content_piece = content_piece.replace('```', '', 1).lstrip()
            first_chunk_skipped = True
        
        # Remove ending code blocks
        if content_piece.endswith('```'):
            content_piece = content_piece.replace('```', '').rstrip()

        return content_piece, inside_think_block, first_chunk_skipped
    
    def generate_advice_stream(self, expenses, language="english", tone="formal"):
        """
        Génère des conseils de dépenses en streaming
        
        Args:
            expenses (list): Liste des dépenses
            language (str): Langue souhaitée
            tone (str): Ton souhaité
            
        Yields:
            str: Morceaux de la réponse générée
        """
        logger.info(f"Generating advice with language: {language}, tone: {tone}")
        
        # Vérifier qu'Ollama est accessible
        if not self.check_ollama_status():
            logger.error("Ollama is not accessible!")
            yield "Error: Ollama service is not available. Please check if the service is running."
            return
        
        # Vérifier que le modèle est disponible
        available_models = self.get_available_models()
        if available_models and self.model_name not in available_models:
            logger.warning(f"Model {self.model_name} not found in available models: {available_models}")
            yield f"Warning: Model {self.model_name} may not be available."
        
        messages = self.build_messages(expenses, language, tone)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9
            }
        }

        def stream_response():
            inside_think_block = False
            first_chunk_skipped = False
            response_started = False

            try:
                logger.info(f"Making request to Ollama: {self.ollama_url}")
                with requests.post(self.ollama_url, json=payload, stream=True, timeout=120) as response:
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
                                        
                                        # Nettoyer le contenu
                                        cleaned_content, inside_think_block, first_chunk_skipped = self.clean_response_content(
                                            content_piece, inside_think_block, first_chunk_skipped
                                        )
                                        
                                        if cleaned_content:
                                            yield cleaned_content
                                        
                                    # Vérifier si la réponse est terminée
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
                        error_msg = f"Ollama API Error: {response.status_code}"
                        if response.text:
                            error_msg += f" - {response.text}"
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
    
    def generate_advice(self, expenses, language="english", tone="formal"):
        """
        Génère des conseils de dépenses (version non-streaming)
        
        Args:
            expenses (list): Liste des dépenses
            language (str): Langue souhaitée
            tone (str): Ton souhaité
            
        Returns:
            str: Conseils générés ou message d'erreur
        """
        try:
            # Collecter tout le contenu du stream
            content_parts = []
            for chunk in self.generate_advice_stream(expenses, language, tone):
                content_parts.append(chunk)
            
            return ''.join(content_parts)
            
        except Exception as e:
            logger.error(f"Error generating advice: {e}")
            return f"Error generating advice: {str(e)}"