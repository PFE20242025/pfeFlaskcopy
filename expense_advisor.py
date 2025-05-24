# expense_advisor.py
import requests
import json
import time

class ExpenseAdvisor:
    def __init__(self):
        self.ollama_url = "http://20.169.88.121:11434/api/chat"
        self.model_name = "phi3"
    
    def build_messages(self, expenses, language, tone):
        # More explicit debug
        print(f"Building messages with language: {language}, tone: {tone}")
        
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
        
        # ADD STRONGER LANGUAGE INSTRUCTION
        system_message = f"IMPORTANT: You must respond in {language} language only and with {tone} tone. Here is a breakdown of my current expenses. Based on this, can you give me recommendations to help me save money. {language_instruction} {tone_instruction}"
        
        user_message = (
            f"My recent expenses are:\n" + "\n".join(f"- {expense}" for expense in expenses) +
            f"\n\nIMPORTANT: YOU MUST RESPOND IN {language.upper() } LANGUAGE ONLY. {language_instruction} {tone_instruction} Format the response in HTML starting from the body tag and give me only recommendations. Don't generate anything else in your response. I want the response to be well structured because I will visualize it directly on my website. Write the expense first, then the recommendation for it, and use the same style for all expenses. Don't use tables."
        )
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Debug: Print complete messages for debugging
        print(f"System message: {system_message}")
        print(f"First 150 chars of user message: {user_message[:150]}...")
        
        return messages
    
    def generate_advice_stream(self, expenses, language="english", tone="formal"):
        # Debug print to verify parameters are received correctly
        print(f"[ExpenseAdvisor] Generating advice with language: {language}, tone: {tone}")
        
        messages = self.build_messages(expenses, language, tone)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True
        }

    def generate_advice(self, expenses, language="english", tone="formal"):
        messages = self.build_messages(expenses, language, tone)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False 
        }
    try:
        response = requests.post(self.ollama_url, json=payload)
        if response.status_code == 200:
            data = response.json()
            # Assuming the full content is in: data['choices'][0]['message']['content']
            # (adjust this based on your actual API response structure)
            full_content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            # Remove any markdown fences if present
            if full_content.startswith("```html"):
                full_content = full_content.replace("```html", "").rstrip("```").strip()
            return full_content
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Connection error: {str(e)}"


        def stream_response():
            inside_think_block = False
            first_chunk_skipped = False  # Flag to fix the first bad piece

            try:
                with requests.post(self.ollama_url, json=payload, stream=True) as response:
                    if response.status_code == 200:
                        for line in response.iter_lines(decode_unicode=True):
                            if line:
                                try:
                                    if isinstance(line, bytes):
                                        line = line.decode('utf-8')
                                    chunk = json.loads(line)

                                    content_piece = chunk.get('message', {}).get('content', '')
                                    if content_piece:
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
                                            content_piece = content_piece.lstrip()  # Remove extra spaces just in case
                                            if content_piece.startswith('```html'):
                                                content_piece = content_piece.replace('```html', '', 1).lstrip()
                                            first_chunk_skipped = True

                                        yield content_piece
                                except Exception as e:
                                    yield f"\n[Streaming Error]: {str(e)}\n"
                    else:
                        yield f"Error: {response.status_code} - {response.text}"
            except Exception as e:
                yield f"\n[Connection Error]: {str(e)}\n"

        return stream_response()
