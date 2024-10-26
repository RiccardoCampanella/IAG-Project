from groq import Groq
import yaml
import os
import re
from collections import defaultdict, Counter
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import connected_components

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

class LLMService:

    def __init__(self):
        self.client = Groq(  # Initialize communication with the large language model
        api_key=config['keys']['llm_api_key'],
        )
        self.model = config['model_specs']['model_type'] 
        self.model_temperature = config['model_specs']['temperature']
        self.showResults = False
        
        self.lstLLMQueries = []

    def get_LLM_queries(self):
        LLMprompt = 'Can you provide a two ways to turn the statement ' + \
        self.statement + \
        ' into a question? Please make the second question a negation of the first without using the word "not". When it is a negation of the statement add "(negation)" to the end. Keep the answer concise.'
        
        LLMresponse = self.LLM_query(LLMprompt)
        if self.showResults: print(LLMresponse)
        lstQuestions = self.extract_text(LLMresponse)
        
        for dictQuestion in lstQuestions:

            llmQuery = dictQuestion["question"] + \
                " Present arguments concisely, " + \
                "focusing on evidence without speculation, " + \
                "and structure the response as evidence for or against the statement. " + \
                "Please give every statement a score from 1-10 on how reliable it is." 
            
            self.lstLLMQueries.append({'query':llmQuery,'isNegated':dictQuestion["isNegated"]})
        
    def extract_information(self, text):
        pattern = re.compile(r'\d+\.\s*(\*\*(.*?)\*\*)?\s*(.*?):\s*(.+?)\s+\((score|Score):\s+(\d+)\/10.*?\)', re.DOTALL)
    
        evidence_for = []
        evidence_against = []

        if text.find("Evidence For") < text.find("Evidence Against"):
        
            for_match = re.search(r"Evidence For(.*?)Evidence Against", text, re.DOTALL)
            if for_match:
                evidence_for = pattern.findall(for_match.group())
                
            # Extract "Evidence Against" and "Evidence Against the Statement" blocks
            against_match = re.search(r"Evidence Against(.*?)Conclusion", text, re.DOTALL)
            if against_match:
                evidence_against = pattern.findall(against_match.group())
        else:
            for_match = re.search(r"Evidence For(.*?)Conclusion", text, re.DOTALL)
            if for_match:
                evidence_for = pattern.findall(for_match.group())
                
            # Extract "Evidence Against" and "Evidence Against the Statement" blocks
            against_match = re.search(r"Evidence Against(.*?)Evidence For", text, re.DOTALL)
            if against_match:
                evidence_against = pattern.findall(against_match.group())
        
        # Build results in the required format
        result = []
        for idx, (bold_title, title, short_title, statement, score_label, score) in enumerate(evidence_for):
            statement_text = f"{(title or short_title).strip()}: {statement.strip()}" if title else statement.strip()
            result.append({
                "score": int(score),
                "text": statement_text,
                "boolCounterArgument": False
            })
        
        for idx, (bold_title, title, short_title, statement, score_label, score) in enumerate(evidence_against):
            statement_text = f"{(title or short_title).strip()}: {statement.strip()}" if title else statement.strip()
            result.append({
                "score": int(score),
                "text": statement_text,
                "boolCounterArgument": True
            })
        return result
    
    def get_LLM_arguments(self):
        self.lstLLMArguments = []
        for query in self.lstLLMQueries:

            LLMresponse = self.LLM_query(query["query"])
            if self.showResults: 
                print(query["query"])
                print(LLMresponse)
            lstArguments = self.extract_information(LLMresponse)

            for dictArgument in lstArguments:
                
                if query["isNegated"] != dictArgument["boolCounterArgument"]:
                    dictArgument["boolCounterArgument"] = True
                else: dictArgument["boolCounterArgument"] = False
                
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

        self.get_LLM_queries()
        self.get_LLM_arguments()
        if self.showResults: 
            for i in self.lstLLMArguments:
                print(i)
        counter_args = [arg for arg in self.lstLLMArguments if not arg["boolCounterArgument"]]
        args = [arg for arg in self.lstLLMArguments if arg["boolCounterArgument"]]
        counter_args = self.compare_arguments(counter_args)
        args = self.compare_arguments(args)
        
        trust = self.get_trust(args+counter_args)
        if self.showResults: 
            print([str(arg["boolCounterArgument"] == False) + " " + str(arg["score"]) for arg in args+counter_args])
            print(trust)
        return args+counter_args, trust
    
    def get_trust(self, args):
        trust = 0.5
        for arg in args:
            if arg["boolCounterArgument"]:
                trust = trust * (1-arg["score"]/10)
            else:
                trust = trust * (1/(1-arg["score"]/10))
        if trust > 1: 
            trust = 0.5
            for arg in args:
                if not arg["boolCounterArgument"]:
                    trust = trust * (1-arg["score"]/10)
                else:
                    trust = trust * (1/(1-arg["score"]/10))
            trust = 1 - trust
        return trust

    
    def LLM_query(self, LLMQuery): # Returns the answer to a given prompt from the LLM
        self.model = config['llm_mapper']['model_type'] 
        self.model_temperature = config['model_specs']['temperature']
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

    def compare_arguments(self, args):

        texts = [arg["text"] for arg in args]

        vectorizer = TfidfVectorizer()
        sentence_vectors = vectorizer.fit_transform(texts)

        # Step 2: Compute cosine similarity
        cosine_sim_matrix = cosine_similarity(sentence_vectors)

        # Step 3: Apply a threshold to create a similarity graph (adjacency matrix)
        threshold = 0.25
        similarity_graph = (cosine_sim_matrix >= threshold).astype(int)

        # Step 4: Find connected components (groups of similar sentences)
        n_components, labels = connected_components(csgraph=similarity_graph, directed=False, return_labels=True)

        # Step 5: Group sentences by their component labels
        grouped_sentences = {}
        for i, label in enumerate(labels):
            if label not in grouped_sentences:
                grouped_sentences[label] = []
            grouped_sentences[label].append(args[i])

        # Get the avg of the grouped sentences
        arg_list = []
        for group, sents in grouped_sentences.items():
            avg_score = sum([sent["score"] for sent in sents])/len(sents)
            new_arg = max(sents, key=lambda x: x["score"])
            new_arg["score"] = avg_score
            new_arg["model"] = "LLM_GROQ"
            arg_list.append(new_arg)
            
        
        return arg_list
            



if __name__ == '__main__':
    LLM = LLMService()
    #
    LLM.query("Swimming is good for your health")
