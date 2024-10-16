from owlready2 import *
from pprint import pprint
path = "C:\\Users\\tijme\\Documents\\GitHub\\IAG-Project\\intelligent_agents.rdf"


class goal_based_agent:
  def __init__(self, path="onto_pizza.owl"):
    self.ontology = get_ontology(path)
    self.ontology.load()
    pprint(list(self.ontology.classes())[0])

    #self.label_to_class = {ent.label[0]: ent for ent in self.ontology.classes()}
    #self.label_to_prop = {prop.label[0]: prop for prop in self.ontology.properties()}

    self.class_to_label = {ent:ent.label for ent in self.ontology.classes()}
    #self.prop_to_label = {prop:prop.label[0] for prop in self.ontology.properties()}

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
