from owlready2 import *
from pprint import pprint
from groq import Groq
path = "C:\\Users\\tijme\\Documents\\GitHub\\IAG-Project\\intelligent_agents.rdf"
protege_path = "C:\\Users\\tijme\\Desktop\\Protege-5.6.4\\jre\\bin\\java.exe"

owlready2.JAVA_EXE = protege_path



class goal_based_agent:
  def __init__(self, path):
    
    self.path = path
    self.ontology = get_ontology(path) # Load the ontology
    self.ontology.load()
    
    with self.ontology: # Run the reasoner to obtain the inferences
      sync_reasoner()

    self.LLM = Groq(  # Initialize communication with the large language model
    api_key=os.environ.get("GROQ_API_KEY"),
    )
    self.model="llama3-8b-8192"
 
  
    self.state = 'Idle'
    self.actions = self.get_action_list()
    self.goals = []
  
    self.confidence = 0
    self.trust = 0
  
  def run(self):
    

  def get_action_list(self):
    action_list = []
    if self.state == "Idle":
      action_list.append(self.await_user)
    
    return action_list

  def await_user(self):
    statement = input("Enter statement...")
    self.state = "Input Processing"
    self.statement = statement
  
  def simple_query(self):
    ar1 = self.ontology.Healthy.instances()
    ar2 = self.ontology.Sport.instances()
    inte = list(set(ar1).intersection(ar2))
    print("result", inte[0])
  
  
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
    )

    print(chat_completion.choices[0].message.content)
    pprint(chat_completion)
    
  #def function_make_query_from_nl():
  
  #def function_get_info_ontology():

  #def function_get_info_gpt():

  #def function_rank_info():

  #def function_make_reccomendation():



agent = goal_based_agent(path)
agent.simple_query()
