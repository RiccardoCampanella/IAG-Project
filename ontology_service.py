from owlready2 import * 
import owlready2
import yaml
from groq import Groq
import os
from functools import reduce
import re
from typing import List, Tuple, Optional, Dict, Any
import requests

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

from dotenv import load_dotenv
load_dotenv()

    
class OntologyService:
    """Service for handling all LLM-related operations."""

    def __init__(self):
        #os.environ["GROQ_API_KEY"] = config['keys']['llm_api_key']
        owlready2.JAVA_EXE = config['paths']['protege_path']
        self.path = config['paths']['ontology_local_path']
        self.ontology = get_ontology(self.path).load() # Load the ontology
        
        try:
            self.ontology = get_ontology(self.path).load()
            with self.ontology:
                sync_reasoner()
            self.use_backup = False
        except Exception as e:
            print(f"Warning: Primary ontology failed to load ({str(e)}). Using Wikipedia backup.")
            self.use_backup = True
            self.ontology = None

    def query(self, arguments, target=None, forall=None):
        """Enhanced query method with Wikipedia fallback."""
        if not self.use_backup:
            return self._primary_query(arguments, target, forall)
        else:
            pass
    
    def _query_dbpedia(self, sparql_query):
        """
        Execute a SPARQL query on DBpedia.
        
        Args:
            sparql_query: The SPARQL query string.
        
        Returns:
            List of results from DBpedia.
        """
        dbpedia_endpoint = "http://dbpedia.org/sparql"
        response = requests.get(dbpedia_endpoint, 
                                params={'query': sparql_query, 'format': 'json'})
        response.raise_for_status()  # Raise an error for bad responses
        return response.json().get('results', {}).get('bindings', [])

    def _build_dbpedia_query(self, datatype, argument):
        """
        Build a SPARQL query for DBpedia based on the datatype and argument.
        
        Args:
            datatype: The type of query (class or objectproperty).
            argument: The argument for the query.
        
        Returns:
            SPARQL query string.
        """
        if datatype == "class":
            return f"""
            SELECT ?instance WHERE {{
                ?instance a <http://dbpedia.org/ontology/{argument}> .
            }}
            """
        elif datatype == "objectproperty":
            property_name, property_value = argument
            return f"""
            SELECT ?instance WHERE {{
                ?instance <http://dbpedia.org/property/{property_name}> <http://dbpedia.org/resource/{property_value}> .
            }}
            """
        return None
    
    def _primary_query(self, arguments, target=None, forall=None):
        """
        Execute a primary query on the Protege ontology.
        
        Args:
            arguments: List of tuples containing (datatype, argument) pairs
            target: Optional target instance to check for
            forall: Optional universal quantifier tuple (type, argument)
            
        Returns:
            list or bool: List of matching instances, or boolean for existence queries
            
        Raises:
            ValueError: If required ontology elements are not found
        """
        instances_list = []
        
        try:
            for datatype, argument in arguments:
                if datatype == "class":
                    some_class = self.ontology.search_one(iri="*#" + argument)
                    if some_class is None:
                        raise ValueError(f"Class not found in ontology: {argument}")
                    class_instances = some_class.instances()
                    if class_instances is None:
                        raise ValueError(f"No instances found for class: {argument}")
                    instances_list.append([str(x) for x in class_instances])
                
                elif datatype == "objectproperty":
                    property_name, property_value = argument
                    sublist = self.searchproperties(property_name, property_value)
                    if not sublist:  # If empty list or None
                        raise ValueError(f"No instances found for property {property_name} with value {property_value}")
                    instances_list.append(sublist)
            
            # If no valid instances were found, return appropriate result
            if not instances_list:
                if target is not None:
                    return False
                return []
            
            # Calculate intersection of all instance lists
            intersection = list(reduce(set.intersection, map(set, instances_list)))
            
            # Handle target checking
            if target is not None:
                return target in intersection
            
            # Handle universal quantifier
            if forall is not None:
                forall_type, forall_arg = forall
                
                if forall_type == "class":
                    forall_class = self.ontology.search_one(iri="*#" + forall_arg)
                    if forall_class is None:
                        raise ValueError(f"Universal quantifier class not found: {forall_arg}")
                    forall_instances = forall_class.instances()
                    if forall_instances is None:
                        raise ValueError(f"No instances found for universal quantifier class: {forall_arg}")
                    return all(i in forall_instances for i in intersection)
                
                elif forall_type == "objectproperty":
                    property_name, property_value = forall_arg
                    object_list = self.searchproperties(property_name, property_value)
                    if object_list is None:
                        raise ValueError(f"No instances found for universal quantifier property {property_name} with value {property_value}")
                    return all(i in object_list for i in intersection)
            
            # If no target or forall, return the intersection
            return intersection

        except ValueError as e:
            print(f"Debug - Query failed: {str(e)}")
            print(f"Debug - Arguments: {arguments}")
            print(f"Debug - Target: {target}")
            print(f"Debug - Forall: {forall}")
            # Attempt to query DBpedia if local ontology query failed
            dbpedia_results = []
            for datatype, argument in arguments:
                dbpedia_query = self._build_dbpedia_query(datatype, argument)
                if dbpedia_query:
                    dbpedia_results += self._query_dbpedia(dbpedia_query)

            # If results are found from DBpedia, process them
            if dbpedia_results:
                return [result['instance']['value'] for result in dbpedia_results]
            print("No results from DB pedia")
            
            raise  # Re-raise if no DBpedia results found
    
    def nlp_to_protege_query(self, natural_language_query):
        """Convert natural language queries into structured queries."""
        query = natural_language_query.lower().strip()
        
        query_arguments = []
        target = None
        forall = None
        
        # Simplified prompt with exact expected YAML format
        prompt = (
            f"Analyze this question: '{query}'\n"
            "Return ONLY a YAML response in this EXACT format with no additional text:\n\n"
            "classes: []\n"
            "properties: []\n"
            "instance: null\n"
            "universal: false\n\n"
            "Fill in the arrays and values based on these rules:\n"
            "- classes: Add class names like Person, Athlete\n"
            "- properties: Add [property, object] pairs like [Eats, Food]\n"
            "- instance: Add specific instance name or keep null\n"
            "- universal: true if asking about 'all', false otherwise")
        
        prompt_ontology_structure = ("Valid classes: [DietaryPreference, Health, Repetitiveness, Availability, \
                            EatingConditions, MovingConditions, Cuisine, Weather, \
                            Nutrient, SportType, CookingMethod, Recipe, Name, \
                            SportName, Season, RecipeName, HealthConditions, \
                            BodyParts, IngredientName, Sport, CommonInjuries, \
                            Quantity, Temperature, Gender, Diseases, Symptoms, \
                            Human, Food, FoodName, Humidity, Ingredients, Climate, \
                            Age, CaloriesBurned, Environment, SourceType, Personal]\
            Valid properties: [\
                causeDisease, conditionHasSymptom, hasAge, belongsToCuisine,\
                requiresFoodAvailability, dependsOnClimate, isPartOfDiet,\
                isMainIngredientIn, causesAllergicReaction, usesBodyPart,\
                causesCondition, hasCondition, affectsFood, influencesDisease,\
                affectsSport, relatesToBodyPart, hasQuantity, hasGender,\
                hasDietaryPreference, impactsCaloriesBurned, containsNutrient,\
                requiresSpecificSeason, eatsRecipe, usesIngredient,\
                isRestrictedByHealthCondition, hasCommonInjuries, isSeasonalIn,\
                hasRepetitiveness, isAvailableIn, hasSymptom, isCompatibleWith,\
                eatsIngredient, isMainSportFor, isSimilarTo, diseaseHasSymptom,\
                performsSport, isMainIngredientOf, limitsFoodIntake,\
                recipeBelongsToCuisine, limitsEatingOptions, usesCookingMethod,\
                restrictsSportParticipation\
            ]\
            Valid individuals: [\
                conditionHasSymptom, requiresFoodAvailability, EatingConditions,\
                Name, FrozenShoulder, GrilledSalmon, IngredientName, hasQuantity,\
                Temperature, containsNutrient, Diseases, eatsRecipe, isSeasonalIn,\
                Pineapple, Ingredients, isCompatibleWith, Snowy, diseaseHasSymptom,\
                Apple, performsSport, Personal, restrictsSportParticipation,\
                DietaryPreference, HotTemperature, dependsOnClimate, isMainIngredientIn,\
                usesBodyPart, CookingMethod, affectsFood, Candy, CommonInjuries,\
                hasGender, hasDietaryPreference, Handball, impactsCaloriesBurned,\
                requiresSpecificSeason, Gender, MuscleWeakness, Symptoms, usesIngredient,\
                Salmon, FoodName, isAvailableIn, Humidity, hasSymptom, Climate,\
                Obesity, SourceType, usesCookingMethod, causeDisease, belongsToCuisine,\
                Availability, MovingConditions, causesCondition, Cycling, Season,\
                HealthConditions, BodyParts, hasCommonInjuries, hasRepetitiveness,\
                Soccer, isSimilarTo, isMainSportFor, CaloriesBurned, isMainIngredientOf,\
                recipeBelongsToCuisine, Pieter, hasAge, Repetitiveness, isPartOfDiet,\
                Pasta, Weightlifting, Cuisine, Weather, Nutrient, SportType,\
                causesAllergicReaction, Rainy, hasCondition, Carbs, SportName,\
                influencesDisease, affectsSport, relatesToBodyPart, RecipeName,\
                Quantity, Peanut, isRestrictedByHealthCondition, OliveOil,\
                GlutenAllergy, eatsIngredient, Age, LossOfAppetite, Salad,\
                limitsFoodIntake, limitsEatingOptions, Vegan\
            ]"
        )
        
        prompt_complete = f"""Analyze this question: '{{query}}'

        Return ONLY a YAML response in this EXACT format with no additional text:

        classes:
        - name: null  # Must be one of the valid classes
            superClass: null  # Parent class if applicable
            subClasses: []  # List of child classes
            properties: []  # List of applicable properties
            restrictions: []  # Any restrictions on the class

        properties:
        - name: null  # Must be one of the valid properties
            domain: null  # Source class
            range: null  # Target class
            type: null  # ObjectProperty or DatatypeProperty

        instances:
        - name: null  # Must be one of the valid individuals
            type: null  # Class this instance belongs to
            properties: []  # List of property-value pairs

        relationships:
        - subject: null  # Source class/instance
            predicate: null  # Property relating subject to object
            object: null  # Target class/instance

        Valid classes: [
            DietaryPreference, Health, Repetitiveness, Availability, 
            EatingConditions, MovingConditions, Cuisine, Weather, 
            Nutrient, SportType, CookingMethod, Recipe, Name, 
            SportName, Season, RecipeName, HealthConditions, 
            BodyParts, IngredientName, Sport, CommonInjuries, 
            Quantity, Temperature, Gender, Diseases, Symptoms, 
            Human, Food, FoodName, Humidity, Ingredients, Climate, 
            Age, CaloriesBurned, Environment, SourceType, Personal
        ]

        Valid properties: [
            causeDisease, conditionHasSymptom, hasAge, belongsToCuisine,
            requiresFoodAvailability, dependsOnClimate, isPartOfDiet,
            isMainIngredientIn, causesAllergicReaction, usesBodyPart,
            causesCondition, hasCondition, affectsFood, influencesDisease,
            affectsSport, relatesToBodyPart, hasQuantity, hasGender,
            hasDietaryPreference, impactsCaloriesBurned, containsNutrient,
            requiresSpecificSeason, eatsRecipe, usesIngredient,
            isRestrictedByHealthCondition, hasCommonInjuries, isSeasonalIn,
            hasRepetitiveness, isAvailableIn, hasSymptom, isCompatibleWith,
            eatsIngredient, isMainSportFor, isSimilarTo, diseaseHasSymptom,
            performsSport, isMainIngredientOf, limitsFoodIntake,
            recipeBelongsToCuisine, limitsEatingOptions, usesCookingMethod,
            restrictsSportParticipation
        ]

        Valid individuals: [
            conditionHasSymptom, requiresFoodAvailability, EatingConditions,
            Name, FrozenShoulder, GrilledSalmon, IngredientName, hasQuantity,
            Temperature, containsNutrient, Diseases, eatsRecipe, isSeasonalIn,
            Pineapple, Ingredients, isCompatibleWith, Snowy, diseaseHasSymptom,
            Apple, performsSport, Personal, restrictsSportParticipation,
            DietaryPreference, HotTemperature, dependsOnClimate, isMainIngredientIn,
            usesBodyPart, CookingMethod, affectsFood, Candy, CommonInjuries,
            hasGender, hasDietaryPreference, Handball, impactsCaloriesBurned,
            requiresSpecificSeason, Gender, MuscleWeakness, Symptoms, usesIngredient,
            Salmon, FoodName, isAvailableIn, Humidity, hasSymptom, Climate,
            Obesity, SourceType, usesCookingMethod, causeDisease, belongsToCuisine,
            Availability, MovingConditions, causesCondition, Cycling, Season,
            HealthConditions, BodyParts, hasCommonInjuries, hasRepetitiveness,
            Soccer, isSimilarTo, isMainSportFor, CaloriesBurned, isMainIngredientOf,
            recipeBelongsToCuisine, Pieter, hasAge, Repetitiveness, isPartOfDiet,
            Pasta, Weightlifting, Cuisine, Weather, Nutrient, SportType,
            causesAllergicReaction, Rainy, hasCondition, Carbs, SportName,
            influencesDisease, affectsSport, relatesToBodyPart, RecipeName,
            Quantity, Peanut, isRestrictedByHealthCondition, OliveOil,
            GlutenAllergy, eatsIngredient, Age, LossOfAppetite, Salad,
            limitsFoodIntake, limitsEatingOptions, Vegan
        ]

        Rules:
        1. Only use elements from the valid classes, properties, and individuals lists
        2. Ensure all relationships use valid properties connecting appropriate classes
        3. Properties must connect compatible domains and ranges
        4. Instances must be of valid classes
        5. All returned elements must be from the provided lists

        Example Usage:
        Query: "What foods are suitable for someone with gluten allergy?"

        Response:
        classes:
        - name: Food
            properties: [isRestrictedByHealthCondition, containsNutrient]
        - name: HealthConditions
            properties: [restrictsFood]

        properties:
        - name: isRestrictedByHealthCondition
            domain: Food
            range: HealthConditions
            type: ObjectProperty

        instances:
        - name: GlutenAllergy
            type: HealthConditions
            properties: [restrictsFood]

        relationships:
        - subject: Food
            predicate: isRestrictedByHealthCondition
            object: GlutenAllergy
        """

        # Usage:
        # query = "What sports are good for weight loss?"
        # response = generate_ontology_response(prompt_template.format(query=query))
        self.client = Groq(  # Initialize communication with the large language model
        api_key=os.environ.get("GROQ_API_KEY"),
        )
        self.model = config['model_specs']['model_type'] 
        self.model_temperature = config['model_specs']['temperature']
        self.showResults = False

        try:
            # Create chat completion
            response = self.client.chat.completions.create(
                model=config['model_specs']['model_type'],
                messages=[{
                    "role": "user",
                    "content": prompt#+prompt_ontology_structure
                }],
                temperature=0.0  # Set to 0 for more consistent formatting
            )
            
            # Get response content
            content = response.choices[0].message.content.strip()
            
            # Directly parse the content as YAML
            try:
                analysis = yaml.safe_load(content)
            except yaml.YAMLError as yaml_err:
                print(f"Debug - Raw LLM response:\n{content}")
                raise ValueError(f"YAML parsing error: {str(yaml_err)}")
                
            if not isinstance(analysis, dict):
                raise ValueError("LLM response is not a valid YAML dictionary")
                
            # Validate required keys
            required_keys = {'classes', 'properties', 'instance', 'universal'}
            if not all(key in analysis for key in required_keys):
                raise ValueError(f"Missing required keys in response. Found: {list(analysis.keys())}")
            
            # Process classes
            if isinstance(analysis.get('classes'), list):
                for class_name in analysis['classes']:
                    if isinstance(class_name, str):
                        query_arguments.append(("class", class_name.capitalize()))
            
            # Process properties
            if isinstance(analysis.get('properties'), list):
                for prop_pair in analysis['properties']:
                    if isinstance(prop_pair, list) and len(prop_pair) == 2:
                        prop, obj = prop_pair
                        if isinstance(prop, str) and isinstance(obj, str):
                            query_arguments.append(
                                ("objectproperty", [prop.capitalize(), obj.capitalize()])
                            )
            
            # Process instance
            instance = analysis.get('instance')
            if isinstance(instance, str):
                target = instance.capitalize()
            
            # Process universal quantifier
            if analysis.get('universal', False) and analysis.get('properties'):
                # Get the last property for universal quantification
                prop_pairs = analysis['properties']
                if prop_pairs:
                    last_prop = prop_pairs[-1]
                    if isinstance(last_prop, list) and len(last_prop) == 2:
                        prop, obj = last_prop
                        forall = ("objectproperty", [prop.capitalize(), obj.capitalize()])
                        # Remove from query arguments if present
                        prop_tuple = ("objectproperty", [prop.capitalize(), obj.capitalize()])
                        if prop_tuple in query_arguments:
                            query_arguments.remove(prop_tuple)
            
        except Exception as e:
            print(f"Debug - Error details: {str(e)}")
            raise ValueError(f"Query processing failed: {str(e)}")
        
        # Determine return format based on query type
        if "who" in query or "what" in query:
            return query_arguments, None, None
        elif "does" in query or "is" in query or "are" in query:
            if target:
                return query_arguments, target, None
            elif analysis.get('universal'):
                return query_arguments, None, forall
        elif "how many" in query:
            return query_arguments, None, None
            
        # Default return if no specific query type is matched
        return query_arguments, None, None

    def searchproperties(self, prop, argument):
        """Search for instances with specified property (works with primary ontology only)."""
        if self.use_backup:
            content = self.wiki_ontology.get_page_content(prop)
            relations = self.wiki_ontology.extract_relations(content, "objectproperty")
            return [rel for rel in relations if argument.lower() in rel.lower()]
            
        objectlist = []
        ontoproperty = self.ontology.search_one(iri="*#" + prop)
        ontoinstance = self.ontology.search_one(iri="*#" + argument)
        
        for instance in self.ontology.individuals():
            if ontoproperty in instance.get_properties():
                for sub, obj in ontoproperty.get_relations():
                    if sub == instance and obj == ontoinstance:
                        objectlist.append(instance)
        
        return objectlist