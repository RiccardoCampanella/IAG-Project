# IAG-Project

### Since the agent requires an LLM key to run we have sent a video to whow off our agent. 

### Should you want to run the agent yourself. Please update the Config file with the required paths and Groq LLM key. 
##### [https://groq.com/]
### After that the agent should run automatically when running expert_reasoner_agent.py


# Quick notes on each python file:
* agent_architecture.py    -  Earlier version of the agent
* arguments_examples.py    -  Examples on how we save the arguments
* config.yaml              -  config
* era_agent_trainer.py     -  Trains the agents hyperparameters based on the accuracy metric
* expert_reasoner_agent.py -  Agent main file. Runs and contains the agent
* intelligent_agents.rdf   -  Earlier version of the ontology
* llm_service.py           -  Code for communication with the LLM
* ontology.rdf             -  Main ontology file
* ontology_service.py      -  Code for communication with the ontology
