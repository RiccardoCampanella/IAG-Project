from groq import Groq
import yaml
import os
import re
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

class LLMService:

    def __init__(self):

        self.client = Groq(  # Initialize communication with the large language model
        api_key=os.environ.get("GROQ_API_KEY"),
        )
        self.model = config['model_specs']['model_type'] 
        self.model_temperature = config['model_specs']['temperature']

        self.lstLLMQueries = []

    def get_LLM_queries(self):
        LLMprompt = 'Can you provide four different ways to turn the statement ' + \
        self.statement + \
        ' into questions? While ensuring two of these are negations. When it is a negation of the statement add "(negation)" to the end. Keep the answer concise.'
        
        LLMresponse = self.LLM_query(LLMprompt)
        lstQuestions = self.extract_text(LLMresponse)
        
        for dictQuestion in lstQuestions:

            llmQuery = dictQuestion["question"] + \
                " Present arguments concisely, " + \
                "focusing on evidence without speculation, " + \
                "and structure the response as evidence for or against the statement. " + \
                "Please give every statement a score from 1-10 on how reliable it is." 
            
            self.lstLLMQueries.append({'query':llmQuery,'isNegated':dictQuestion["isNegated"]})
        
    def extract_information(self, text):

        pattern = re.compile(r'(\d+)\.\s+(.+?)\s+\(Score:\s+(\d+)\/10\)', re.DOTALL)
        
        evidence_for = []
        evidence_against = []
        
        # Extract "Evidence For" block
        for_match = re.search(r"\*\*Evidence For:\*\*(.*?)\*\*Evidence Against:\*\*", text, re.DOTALL)
        if for_match:
            evidence_for = pattern.findall(for_match.group(1))
        
        # Extract "Evidence Against" block
        against_match = re.search(r"\*\*Evidence Against:\*\*(.*?)\*\*Conclusion:\*\*", text, re.DOTALL)
        if against_match:
            evidence_against = pattern.findall(against_match.group(1))
        
        # Build results in the required format
        result = []
        for idx, (counter, statement, score) in enumerate(evidence_for):
            result.append({
                "score": int(score),
                "text": statement.strip(),
                "boolCounterArgument": False
            })
        
        for idx, (counter, statement, score) in enumerate(evidence_against):
            result.append({
                "score": int(score),
                "text": statement.strip(),
                "boolCounterArgument": True
            })
        return result
    
    def get_LLM_arguments(self):
        self.lstLLMArguments = []
        for query in self.lstLLMQueries:

            LLMresponse = self.LLM_query(query["query"])
            lstArguments = self.extract_information(LLMresponse)
            print(LLMresponse)
            for dictArgument in lstArguments:
                if query["isNegated"] != dictArgument["boolCounterArgument"]:
                    dictArgument["boolCounterArgument"] = False
                else: dictArgument["boolCounterArgument"] = True
                print(dictArgument)
                self.lstLLMArguments.append(dictArgument)
            

    def extract_text(self, text): #Function by chatgpt to parse the query response.
        # Split the text into lines
        lines = text.split("\n")
        
        # Extract questions, removing additional text or brackets
        questions = []
        for line in lines:
            # Check if line contains a question
            if '. ' in line:
                # Extract the question part before any bracketed text
                if '(negation)' in line: isNegated = True
                else: isNegated = False

                question = re.sub(r"\s*\(.*?\)", "", line.split('. ', 1)[1]).strip()
                questions.append({"question":question,"isNegated":isNegated})
        return questions


    def query(self, current_news_item):
        self.statement = current_news_item
        print(self.statement)
        self.get_LLM_queries()
        self.get_LLM_arguments()
        print(self.lstLLMArguments)
        self.compare_arguments()
    
    def LLM_query(self, LLMQuery): # Returns the answer to a given prompt from the LLM
        if type(LLMQuery) != str: return "ERROR! LLMQuery should be a string"
        chat_completion = self.client.chat.completions.create(
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
        return chat_completion.choices[0].message.content

    def compare_arguments(self):

        lenArguments = len(self.lstLLMArguments)
        for i in range(lenArguments-1):
            for j in range(i+1,lenArguments):

                texts = [self.lstLLMArguments[i]["text"],self.lstLLMArguments[j]["text"]]
                vectorizer = CountVectorizer().fit_transform(texts)
                vectors = vectorizer.toarray()

                cos_sim = cosine_similarity(vectors)[0][1]

                print(self.lstLLMArguments[i])
                print(self.lstLLMArguments[j])

                print(cos_sim)
            
           
            



if __name__ == '__main__':
    LLM = LLMService()
    #
     
    LLM.query("Does swimming increase heart attacks?")
