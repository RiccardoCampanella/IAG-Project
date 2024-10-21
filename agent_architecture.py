from owlready2 import *
from pprint import pprint
from groq import Groq
import yaml
import os
import re

# Load the config.yaml file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Intitialize global variables
os.environ["GROQ_API_KEY"] = config['keys']['llm_api_key']
owlready2.JAVA_EXE = config['paths']['protege_path']

class goal_based_agent:
  def __init__(self):
    self.path = config['paths']['ontology_local_path']
    self.ontology = get_ontology(self.path) # Load the ontology
    self.ontology.load()
    
    with self.ontology: # Run the reasoner to obtain the inferences
      sync_reasoner()

    self.LLM = Groq(  # Initialize communication with the large language model
    api_key=os.environ.get("GROQ_API_KEY"),
    )
    self.model = config['model_specs']['model_type'] 
    self.model_temperature = config['model_specs']['temperature'] 
    self.statement = None
    self.state = 'Idle'
    self.actions = []
    self.get_action_list()
    self.goals = []
  
    self.confidence = 0
    self.trust = 0
    self.boolLLMQuery = False
    self.boolOntologyQuery = True
    self.lstLLMQueries = []
  
  def run(self):
    while self.actions != []:
      print([action.__name__ for action in self.actions],self.state)
      self.actions[0]()
      self.actions.pop(0)
      if self.actions == []:
        self.get_action_list()
      self.state_transition()    
      
  def state_transition(self):
    print(self.lstLLMQueries != [])
    if self.state == "Idle" and self.statement != None:
      self.state = "Input Processing"
    elif self.state == "Input Processing" and self.lstLLMQueries != []:
      self.state = "Information Gathering"  
    elif self.state == "Information Gathering":
      self.state = "End"
    
  def get_action_list(self):
    action_list = []
    if self.state == "Idle":
      action_list.append(self.await_user)
    elif self.state == "Input Processing":
      action_list.append(self.simple_query)
      action_list.append(self.get_LLM_queries)
    elif self.state == "Information Gathering":
      action_list.append(self.get_LLM_arguments)
    elif self.state == "End":
      return
    self.actions = action_list

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
    print(self.lstLLMQueries)
    self.boolGetLLMQueries = True

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

  def await_user(self): # Wait untill the user inputs a statement. This switches agent state to 'Input Processing'
    if self.statement != None: return 
    statement = input("Enter statement...")
    self.statement = statement
    
  def simple_query(self): # Simple query for testing
    ar1 = self.ontology.Healthy.instances()
    ar2 = self.ontology.Sport.instances()
    inte = list(set(ar1).intersection(ar2))
    print("result", inte[0])
    
  
  def LLM_query(self,LLMQuery): # Returns the answer to a given prompt from the LLM
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
 
  """ # Pseudo code for potential ontology query 

  statement: 'Does eating spicy food cause hair loss'

  structure: (Domain Action Range) implies (Domain attribute Range2)
    ObjProperty = eating
    Range = spicy food
    Domain = ??
    Range2 = hair loss

    
  statement: swimming is good for your heart

  structure: (Domain ObjProperty Range) implies (Domain objProperty Range2)

  ObjProperty = ??
  Range = swimming
  Domain = ??
  
  Range2 = Heart


  def find_object_properties():
    


  
  
  
  """


agent = goal_based_agent()
agent.statement = 'Does eating spicy food cause hair loss'
agent.run()
