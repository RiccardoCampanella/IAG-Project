# IAG-Project

### Since the agent requires an LLM key to run we have sent a video to show off our agent. 

### Should you want to run the agent yourself. Please update the Config file with the required paths and Groq LLM key. 
##### [https://groq.com/]
### After that the agent should run automatically when running expert_reasoner_agent.py
#### When running the agent a log directory is created containing the system logs automatically generated when running a code of the related class.
#### When training the agent (era_agent_trainer), a training_current_date.json file is generated to save the performances of the agent during the training.

# Quick notes on each python file:
* arguments_examples.py    -  Examples on how we save the arguments
* config.yaml              -  config
* era_agent_trainer.py     -  Trains the agents hyperparameters based on the accuracy metric
* expert_reasoner_agent.py -  Agent main file. Runs and contains the agent
* intelligent_agents.rdf   -  Earlier version of the ontology
* llm_service.py           -  Code for communication with the LLM
* ontology_service.py      -  Code for communication with the ontology
* ontology.rdf             -  Main ontology file
* training_results_20_samples.json - Summay of the performance metrics used in the report
* Fake.csv, True.csv - Datasets used for testing the agent

