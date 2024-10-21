from groq import Groq
import yaml
import os
import re
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

class LLMService:
    """Service for handling all LLM-related operations."""

    def __init__(self):

        self.client = Groq(  # Initialize communication with the large language model
        api_key=os.environ.get("GROQ_API_KEY"),
        )
        self.model = config['model_specs']['model_type'] 
        self.model_temperature = config['model_specs']['temperature']

    def get_LLM_queries(self):
        LLMprompt = 'Can you provide four different ways to turn the statement ' + \
        self.statement + \
        ' into questions, ensuring two of these are negations? Keep the answer concise.'
        
        LLMresponse = self.LLM_query(LLMprompt)
        lstQuestions = self.extract_text(LLMresponse)
        
        for strQuestion in lstQuestions:

            llmQuery = strQuestion + \
                " Present arguments concisely, " + \
                "focusing on evidence without speculation, " + \
                "and structure the response as evidence for or against the statement. " + \
                "Please give every statement a score from 1-10 on how reliable it is." 
            
            self.lstLLMQueries.append(llmQuery)
        

    def get_LLM_arguments(self):
        LLMresponse = self.LLM_query(self.lstLLMQueries[0])
        self.lstArguments = self.extract_text(LLMresponse)
        print(self.lstArguments)
    
    def extract_text(self, text): #Function by chatgpt to parse the query response.
        # Split the text into lines
        lines = text.split("\n")
        
        # Extract questions, removing additional text or brackets
        questions = []
        for line in lines:
            # Check if line contains a question
            if '. ' in line:
                # Extract the question part before any bracketed text
                question = re.sub(r"\s*\(.*?\)", "", line.split('. ', 1)[1]).strip()
                questions.append(question)
        return questions


    def query(self, current_news_item):
        self.statement = current_news_item
        self.get_LLM_queries()
        self.get_LLM_arguments()

    
    def LLM_query(self, LLMQuery): # Returns the answer to a given prompt from the LLM
        if type(LLMQuery) != str: return "ERROR! LLMQuery should be a string"
        chat_completion = self.LLM.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": LLMQuery,
            }
        ],
        model=self.model,
        temperature=self.model_temperature,
        )
        #TODO remove debug print
        print(chat_completion.choices[0].message.content)
        return chat_completion.choices[0].message.content

    def generate_reasoning():
        """Generate reasoning based on evidence."""
        # helpers methods must be private if we want to hide the logic fro the agent
        pass
    
    def evaluate_news_item():
        """Evaluate claims against evidence."""
        # helpers methods must be private if we want to hide the logic fro the agent
        pass