from owlready2 import *
from pprint import pprint
from groq import Groq
import yaml
import os

# Load the config.yaml file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

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
    self.statement = None
    self.state = 'Idle'
    self.actions = []
    self.get_action_list()
    self.goals = []
  
    self.confidence = 0
    self.trust = 0
    self.boolLLMQuery = False
    self.boolOntologyQuery = True
  
  def run(self):
    
    while self.actions != []:
      print([action.__name__ for action in self.actions],self.state)
      self.actions[0]()
      self.actions.pop(0)
      self.state_transition()
      self.get_action_list()
      
  def state_transition(self):
    if self.state == "Idle" and self.statement != None:
      self.state = "Input Processing"
    elif self.state == "Input Processing" and self.boolLLMQuery == True and self.boolOntologyQuery == True:
      self.state = "End"
    
  def get_action_list(self):
    action_list = []
    if self.state == "Idle":
      action_list.append(self.await_user)
    elif self.state == "Input Processing":
      action_list.append(self.simple_query)
      action_list.append(self.get_LLM_queries)
    elif self.state == "End":
      return
    self.actions = action_list

  def get_LLM_queries(self):
    LLMprompt = 'Can you find 5 different ways to ask the following statement as a question: ' + \
    self.statement

    response = self.LLM_query(LLMprompt)
    
    self.lstLLMQueries = "Present arguments concisely, focusing on evidence without speculation, " + \
      "and structure the response as evidence for or against the statement.Present arguments concisely, " + \
        "focusing on evidence without speculation, and structure the response as evidence for or against the statement."
    
    self.boolLLMQuery = True

  def await_user(self):
    if self.statement != None: return 
    statement = input("Enter statement...")
    self.state = "Input Processing"
    self.statement = statement
    
  def simple_query(self):
    ar1 = self.ontology.Healthy.instances()
    ar2 = self.ontology.Sport.instances()
    inte = list(set(ar1).intersection(ar2))
    print("result", inte[0])
    self.state = "End"
  
  def LLM_query(self,LLMQuery):
    if type(LLMQuery) != str: return "ERROR! LLMQuery should be a string"
    chat_completion = self.LLM.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": LLMQuery,
        }
    ],
    model=self.model,
    temperature=0.1,
    )

    print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content
    
  #def function_make_query_from_nl():
  
  #def function_get_info_ontology():

  #def function_get_info_gpt():

  #def function_rank_info():

  #def function_make_reccomendation():



agent = goal_based_agent()
agent.statement = 'Does eating spicy food cause hair loss'
agent.run()
