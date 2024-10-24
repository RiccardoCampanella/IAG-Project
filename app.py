from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import ollama
from answerfromweb import get_answer
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('factchecker.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class FactCheckingLLM:
    def __init__(self):
        logger.info("Initializing fact-checking system...")
        self.model = "llama3:latest"
        try:
            # Test connection with simple prompt
            ollama.chat(model=self.model, messages=[{"role": "user", "content": "test"}])
            logger.info("AI model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading AI model: {str(e)}")
            raise

    def verify_claim(self, claim: str) -> dict:
        """Verify a claim and return structured results"""
        logger.info(f"Starting verification for claim: {claim}")
        
        try:
            # Generate initial analysis
            analysis = self._get_initial_analysis(claim)
            if not analysis:
                raise ValueError("Failed to get initial analysis")

            # Generate and process verification questions
            verifications = self._process_verification_questions(claim)
            
            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "claim": claim,
                "verdict": analysis.get("verdict", "unknown"),
                "reasons": analysis.get("reasons", [])
            }
            
        except Exception as e:
            logger.error(f"Error verifying claim: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "claim": claim
            }

    def _get_initial_analysis(self, claim: str) -> dict:
        """Get initial analysis from the model"""
        try:
            prompt = (
                f"Analyze this claim concisely: '{claim}'\n"
                f"Provide a verdict (pants-fire/false/barely-true/half-true/mostly-true/true) "
                f"and exactly three short reasons. Format as:\n"
                f"Verdict: [verdict]\n"
                f"1. [reason1]\n2. [reason2]\n3. [reason3]"
            )
            
            response = ollama.chat(model=self.model, 
                                messages=[{"role": "user", "content": prompt}])
            
            return self._parse_response(response['message']['content'])
            
        except Exception as e:
            logger.error(f"Error in initial analysis: {str(e)}")
            return {}

    def _process_verification_questions(self, claim: str) -> list:
        """Process verification questions and get answers"""
        verifications = []
        try:
            questions = self._generate_questions(claim)
            for question in questions:
                answer = get_answer(question)
                if answer:
                    verifications.append({
                        "question": question,
                        "answer": answer
                    })
        except Exception as e:
            logger.error(f"Error in verification process: {str(e)}")
        
        return verifications

    def _generate_questions(self, claim: str) -> list:
        """Generate verification questions"""
        try:
            prompt = f"Generate 3 short, factual questions to verify: {claim}"
            response = ollama.chat(model=self.model, 
                                messages=[{"role": "user", "content": prompt}])
            
            questions = [q.strip('123.- ') 
                        for q in response['message']['content'].split('\n') 
                        if q.strip()]
            return questions[:3]  # Ensure we only return 3 questions
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return []

    def _parse_response(self, response: str) -> dict:
        """Parse the model's response into structured format"""
        try:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            verdict = ""
            reasons = []
            
            for line in lines:
                if line.lower().startswith('verdict:'):
                    verdict = line.split(':', 1)[1].strip()
                elif any(line.startswith(str(i)) for i in ['1', '2', '3']):
                    reason = line[2:].strip()
                    if reason:
                        reasons.append(reason)
            
            return {
                "verdict": verdict,
                "reasons": reasons[:3]  # Ensure we only return 3 reasons
            }
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return {"verdict": "", "reasons": []}

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/v1/verify', methods=['POST'])
def verify_claim():
    """Verify a single claim"""
    try:
        data = request.get_json()
        
        if not data or 'claim' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'claim' in request body"
            }), 400
        
        claim = str(data['claim']).strip()
        if not claim:
            return jsonify({
                "success": False,
                "error": "Empty claim provided"
            }), 400
        
        result = fact_checker.verify_claim(claim)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in /verify endpoint: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

# Initialize fact checker
fact_checker = None
try:
    fact_checker = FactCheckingLLM()
except Exception as e:
    logger.error(f"Failed to initialize fact checker: {str(e)}")

if __name__ == '__main__':
    if fact_checker is None:
        print("Error: Failed to initialize fact checker. Please check the logs.")
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)