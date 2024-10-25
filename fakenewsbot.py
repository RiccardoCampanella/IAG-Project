import json
import ollama
from answerfromweb import get_answer
import time

class FactCheckingLLM:
    def __init__(self):
        print("Initializing fact-checking system...")
        self.model = "llama3:latest"
        try:
            ollama.chat(model=self.model, messages=[{"role": "user", "content": "Hello"}])
            print("✓ AI model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading AI model: {str(e)}")
            raise
    
    def verify_claim(self, claim):
        print("\n🔍 Starting fact-check analysis...")
        print("⏳ Generating initial analysis...")
        
        prompt = f"""
        Analyze this claim: "{claim}"
        
        Provide exactly three key verification points and a final verdict using only these labels: 
        pants-fire, false, barely-true, half-true, mostly-true, or true.
        
        Format your response as:
        Verdict: [verdict]
        Reasons:
        1. [first key point]
        2. [second key point]
        3. [third key point]
        """
        
        # Get initial analysis
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        
        # External verification process
        print("🌐 Gathering additional verification from external sources...")
        verification_questions = self.generate_verification_questions(claim)
        for i, question in enumerate(verification_questions, 1):
            if i==4:
                break
            print(f"  • Verifying point {i}/3: {question}")
            external_answer = get_answer(question)
            if external_answer:
                print(f"    ✓ Found relevant information")
                prompt = f"\nAdditional verification: {external_answer}"+prompt
                response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
            else:
                print(f"    ℹ️ No additional information found")
            
        print("✨ Finalizing analysis...")
        return response['message']['content']
    
    def generate_verification_questions(self, claim):
        print("📝 Breaking down claim into verification points...")
        prompt = f"Generate three specific factual questions to verify this claim: {claim}"
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        questions = response['message']['content'].split('\n')
        return [q.strip('123. ') for q in questions if q.strip()]

def main():
    print("\n=== Welcome to the AI Fact Checker! ===")
    print("Type 'bye' to exit the program")
    print("---------------------------------------")
    
    try:
        fact_checker = FactCheckingLLM()
        
        while True:
            print("\n🤔 Waiting for your claim...")
            user_input = input("\nEnter a claim to fact-check: ").strip()
            
            if user_input.lower() == 'bye':
                print("\n👋 Thank you for using AI Fact Checker. Goodbye!")
                break
                
            if not user_input:
                print("❌ Please enter a valid claim.")
                continue
                
            try:
                print("\n🎯 Processing your claim:", user_input)
                result = fact_checker.verify_claim(user_input)
                print("\n📊 Fact Check Result:")
                print("------------------------")
                print(result)
                print("------------------------")
            except Exception as e:
                print(f"❌ Error during fact-checking: {str(e)}")
                print("Please try again with a different claim.")
                
    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        print("Please restart the program.")

if __name__ == "__main__":
    main()