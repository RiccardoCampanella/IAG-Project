from owlready2 import *
from pprint import pprint
path = "C:\\Users\\tijme\\Documents\\GitHub\\IAG-Project\\intelligent_agents.rdf"

from owlready2 import * 
import owlready2
owlready2.JAVA_EXE = r"C:\Users\lucmi\Downloads\Protege-5.6.4-win\Protege-5.6.4\jre\bin\java.exe"
# Create a Graph \Documents\Opdrachten\IAG-Project
onto = get_ontology(r"C:\Users\lucmi\Documents\Opdrachten\IAG-Project\intelligent_agents.rdf").load()
#print(type(onto))
#for entity in onto.entities():
#    print(entity)
with onto:
    sync_reasoner()
#for cls in onto.classes():
 #   print("yeah", cls)
ar1 = onto.Healthy.instances()
ar2 = onto.Sport.instances()
inte = list(set(ar1).intersection(ar2))
print("result", inte)

class goal_based_agent:
  def __init__(self, path="onto_pizza.owl"):
    self.ontology = get_ontology(path)
    self.ontology.load()
    pprint(list(self.ontology.classes())[0])

    #self.label_to_class = {ent.label[0]: ent for ent in self.ontology.classes()}
    #self.label_to_prop = {prop.label[0]: prop for prop in self.ontology.properties()}

<<<<<<< Updated upstream
    self.class_to_label = {ent:ent.label for ent in self.ontology.classes()}
    #self.prop_to_label = {prop:prop.label[0] for prop in self.ontology.properties()}
=======
      llmQuery = strQuestion + \
        " Present arguments concisely, " + \
        "focusing on evidence without speculation, " + \
        "and structure the response as evidence for or against the statement. " + \
        "Please give every statement a score from 1-10 on how reliable it is." 
      
      self.lstLLMQueries.append(llmQuery)
    print(self.lstLLMQueries)
    self.boolGetLLMQueries = True
  
  def reasoning(self, arguments, target = None, forall = None):
      instances_list = []
      for datatype, argument in arguments:
          if datatype == "class":
              some_class = self.ontology.search_one(iri="*#" + argument)
              instances_list.append([str(x) for x in some_class.instances()])
          if datatype == "objectproperty":
              sublist = self.searchproperties(argument[0], argument[1])
              instances_list.append(sublist)
      intersection = list(reduce(set.intersection, map(set, instances_list)))
      if target is not None and target in intersection:
          return True
      if forall is not None:
          if forall[0] == "class":
              forallargument = self.ontology.search_one(iri="*#" + forall[1])
              for i in intersection:
                  if i not in forallargument.instances():
                      return False
              return True
          if forall[0] == "objectproperty":
              objectlist = self.searchproperties(forall[1][0], forall[1][1])
              for i in intersection:
                  if i not in objectlist:
                      return False
              return True
  def searchproperties(self, prop, argument):
      objectlist = []
      ontoproperty = self.ontology.search_one(iri="*#" + prop)
      ontoinstance = self.ontology.search_one(iri="*#" + argument)
      for instance in self.ontology.individuals():
          if ontoproperty in instance.get_properties():
              for sub, obj in ontoproperty.get_relations():
                  if sub == instance and obj == ontoinstance:
                      objectlist.append(instance)
      return objectlist
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
>>>>>>> Stashed changes

    pprint(type(list(self.ontology.classes())[0]))
    list(default_world.sparql("""
           SELECT ?y
           { ?x rdfs:label "Healthy" .
             ?x rdfs:subClassOf* ?y }
    """))
    # Run the reasoner to obtain the inferences
    #with self.ontology:
     #   sync_reasoner(infer_property_values=True)
  #def __init__(self):


  x = 1
  #def function_make_query_from_nl():
  
 

  #def function_get_info_ontology():
  

  #def function_get_info_gpt():

  #def function_rank_info():

  #def function_make_reccomendation():


trust_llm = 6
trust_ont = 8


agent = goal_based_agent(path)
