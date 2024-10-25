from owlready2 import * 
import owlready2
import yaml
from groq import Groq
import os
from functools import reduce
class OntologyService:
    """Service for handling all LLM-related operations."""

    def __init__(self, client):
        self.client = client
        os.environ["GROQ_API_KEY"] = config['keys']['llm_api_key']
        owlready2.JAVA_EXE = config['paths']['protege_path']
        self.path = config['paths']['ontology_local_path']
        #self.ontology = get_ontology(r"C:\Users\lucmi\Documents\Opdrachten\IAG-Project\intelligent_agents.rdf").load()
        self.ontology = get_ontology(self.path).load() # Load the ontology
        #self.ontology.load()
        
        with self.ontology: # Run the reasoner to obtain the inferences
            sync_reasoner()
        #query = [["class", "Healthy"], ["class", "Sport"]]
        query = [["objectproperty", ["Eats", "Cookie"]]]
        self.reasoning(query)

    def query(self, protege_query):
        """Query Ontology for factual knowledge."""
        # helpers methods must be private if we want to hide the logic fro the agent
        pass

    def generate_reasoning():
        """Generate reasoning based on evidence."""
        # helpers methods must be private if we want to hide the logic fro the agent
        pass
    
    def evaluate_news_item():
        """Evaluate claims against evidence."""
        # helpers methods must be private if we want to hide the logic fro the agent
        pass

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