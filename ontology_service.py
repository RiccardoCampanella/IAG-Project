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

    def nlp_to_protege_query(self, natural_language_query):
        """
        Convert natural language queries into structured Protege queries.
        
        Args:
            natural_language_query (str): The natural language question from the user
            
        Returns:
            tuple: (query_arguments, target, forall)
                - query_arguments: List of tuples [(datatype, argument)]
                - target: Optional instance to check for
                - forall: Optional condition tuple (type, value)
                
        Examples:
            "Who are all healthy people?" ->
                ([("class", "Healthy"), ("class", "Person")], None, None)
            "Does John eat cookies?" ->
                ([("objectproperty", ["Eats", "Cookie"])], "John", None)
            "Do all athletes play sports?" ->
                ([("class", "Athlete")], None, ("objectproperty", ["Plays", "Sport"]))
        """
        # Convert query to lowercase for easier processing
        query = natural_language_query.lower().strip()
        
        # Initialize return values
        query_arguments = []
        target = None
        forall = None
        
        # Extract key components using the LLM
        prompt = f"""Analyze this question: "{query}"
        1. Identify classes mentioned (e.g., Person, Athlete)
        2. Identify object properties (e.g., Eats, Plays)
        3. Identify specific instances (e.g., John, Pizza)
        4. Determine if it's asking about "all" of something
        Format response as YAML:
        classes: [list of classes]
        properties: [list of [property, object] pairs]
        instance: specific instance or null
        universal: true/false"""
        
        response = self.client.chat.completions.create(
            model="claude-3-opus-20240229",
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # Parse LLM response
        try:
            analysis = yaml.safe_load(response.choices[0].message.content)
            
            # Add class conditions
            if analysis.get('classes'):
                for class_name in analysis['classes']:
                    query_arguments.append(("class", class_name.capitalize()))
            
            # Add property conditions
            if analysis.get('properties'):
                for prop, obj in analysis['properties']:
                    query_arguments.append(
                        ("objectproperty", [prop.capitalize(), obj.capitalize()])
                    )
            
            # Handle specific instance queries
            if analysis.get('instance'):
                target = analysis['instance'].capitalize()
            
            # Handle universal quantification ("all")
            if analysis.get('universal') and analysis.get('properties'):
                # Take the last property as the universal condition
                prop, obj = analysis['properties'][-1]
                forall = ("objectproperty", [prop.capitalize(), obj.capitalize()])
                # Remove it from query_arguments if it was added
                if ("objectproperty", [prop.capitalize(), obj.capitalize()]) in query_arguments:
                    query_arguments.remove(
                        ("objectproperty", [prop.capitalize(), obj.capitalize()])
                    )
        
        except (yaml.YAMLError, AttributeError, IndexError) as e:
            raise ValueError(f"Failed to parse LLM response: {e}")
        
        # Apply query patterns based on question type
        if "who" in query or "what" in query:
            # Return all matching instances
            return query_arguments, None, None
        
        elif "does" in query or "is" in query or "are" in query:
            # Return boolean result
            if target:
                return query_arguments, target, None
            elif analysis.get('universal'):
                return query_arguments, None, forall
                
        elif "how many" in query:
            # Return count of matching instances
            return query_arguments, None, None
            
        return query_arguments, target, forall

    def _extract_query_components(self, query):
        """
        Helper method to extract basic components from a query string.
        
        Args:
            query (str): Lowercase query string
            
        Returns:
            dict: Extracted components (question_type, keywords, etc.)
        """
        components = {
            'question_type': None,
            'keywords': set(),
            'is_universal': False
        }
        
        # Identify question type
        question_starters = {
            'who': 'entity',
            'what': 'entity',
            'does': 'boolean',
            'is': 'boolean',
            'are': 'boolean',
            'how many': 'count'
        }
        
        for starter, q_type in question_starters.items():
            if query.startswith(starter):
                components['question_type'] = q_type
                break
        
        # Extract keywords
        keywords = set(query.split()) - {
            'who', 'what', 'does', 'is', 'are', 'how', 'many', 'the', 'a', 'an'
        }
        components['keywords'] = keywords
        
        # Check for universal quantification
        components['is_universal'] = 'all' in query or 'every' in query
        
        return components

    def query(self, arguments, target=None, forall=None):
        """
        Query the ontology based on provided arguments and conditions.

        :param arguments: A list of tuples, where each tuple contains a datatype
                        (e.g., "class" or "objectproperty") and the corresponding
                        argument for that datatype.
        :param target: An optional target value that, if present, will be checked 
                    against the intersection of instances obtained from the query.
        :param forall: An optional condition that, if specified, requires that 
                    all instances in the intersection satisfy this condition.
        :return: Returns True if the target is found in the intersection, or if 
                all conditions specified by forall are satisfied; otherwise returns False.
        """
        
        # Initialize an empty list to store instances from the ontology
        instances_list = []
        
        # Iterate over the provided arguments
        for datatype, argument in arguments:
            # Check if the datatype is a class
            if datatype == "class":
                # Search for the class in the ontology using its IRI
                some_class = self.ontology.search_one(iri="*#" + argument)
                # Append the list of instance strings for this class to instances_list
                instances_list.append([str(x) for x in some_class.instances()])
            
            # Check if the datatype is an object property
            if datatype == "objectproperty":
                # Retrieve instances associated with the specified object property
                sublist = self.searchproperties(argument[0], argument[1])
                # Append the sublist of instances to instances_list
                instances_list.append(sublist)
        
        # Calculate the intersection of all instance sets from instances_list
        intersection = list(reduce(set.intersection, map(set, instances_list)))
        
        # If a target is specified and is found in the intersection, return True
        if target is not None and target in intersection:
            return True 
        
        # Check if a forall condition is provided
        if forall is not None:
            # Handle the case where forall is a class
            if forall[0] == "class":
                # Search for the specified class in the ontology
                forallargument = self.ontology.search_one(iri="*#" + forall[1])
                # Check if each instance in the intersection is also in forallargument
                for i in intersection:
                    if i not in forallargument.instances():
                        return False
                # If all instances satisfy the condition, return True
                return True
            
            # Handle the case where forall is an object property
            if forall[0] == "objectproperty":
                # Retrieve the list of instances associated with the specified object property
                objectlist = self.searchproperties(forall[1][0], forall[1][1])
                # Check if each instance in the intersection is also in objectlist
                for i in intersection:
                    if i not in objectlist:
                        return False
                # If all instances satisfy the condition, return True
                return True

    def searchproperties(self, prop, argument):
        """
        Search for instances in the ontology that have a specified property
        linking them to a specified argument.

        :param prop: The name of the property to search for in the ontology.
        :param argument: The name of the argument (instance) to which the property
                        should link the instances.
        :return: A list of instances that have the specified property linking 
                them to the argument.
        """
        
        # Initialize an empty list to store instances that match the search criteria
        objectlist = []
        
        # Search for the specified property in the ontology using its IRI
        ontoproperty = self.ontology.search_one(iri="*#" + prop)
        
        # Search for the specified argument in the ontology using its IRI
        ontoinstance = self.ontology.search_one(iri="*#" + argument)
        
        # Iterate over all individuals (instances) in the ontology
        for instance in self.ontology.individuals():
            # Check if the current instance has the specified property
            if ontoproperty in instance.get_properties():
                # Get the relations associated with the property
                for sub, obj in ontoproperty.get_relations():
                    # Check if the subject is the current instance and the object 
                    # is the specified argument instance
                    if sub == instance and obj == ontoinstance:
                        # If both conditions are met, add the instance to the result list
                        objectlist.append(instance)
        
        # Return the list of instances that have the specified property linking 
        # them to the specified argument
        return objectlist

    
    def evaluate_news_item():
        """Evaluate claims against evidence."""
        # helpers methods must be private if we want to hide the logic fro the agent
        pass