class AgentState:
    IDLE = 0
    INPUT_PROCESSING = 1
    INFORMATION_GATHERING = 2
    EVIDENCE_GATHERING = 3
    REASONING = 4
    RECOMMENDATION_FORMULATION = 5
    OUTPUT_GENERATION = 6
    SELF_EVALUATION = 7

class Goal:
    def __init__(self, description, applicable_states):
        self.description = description
        self.applicable_states = applicable_states

class FakeNewsAgent:
  def __init__(self):
      self.state = AgentState.IDLE
      self.main_goal = "Perfrom fact-checking and provide return statement"
      self.subgoal = [
          Goal("Query the ontology to get evidences", AgentState.EVIDENCE_GATHERING),
          Goal("Retrieve evidences from external source (LLM)", AgentState.EVIDENCE_GATHERING),
          Goal("Compare and analyze the evidences", AgentState.REASONING),
          Goal("Compute and analyze evidences", AgentState.REASONING),
          Goal("Present results to the user")
      ]
      self.current_strategy = None

  def pursue_goal(self, news_item):
      self.analyze_item(news_item)
      while not self.goal_achieved():
          self.select_strategy()
          self.execute_strategy()
          self.evaluate_progress()
          self.adapt_approach()
  
  def analyze_item():
      # code
      return 
  
  def select_strategy():
      # code
      return 
  
  def execute_strategy():
      # code
      return 
  
  def evaluate_strategy():
      # code 
      return 
  
  def evaluate_progress():
      # code
      return 
  
  def adapt_approach():
      # code
      return 1
  
  def goal_achieved():
      # code
      return 
  
  ### Agent-KnowledgeBase interaction functions

  def function_make_query_from_nl():
      # code
      return

  def function_get_info_ontology():
      # code
      return

  def function_get_info_gpt():
      # code
      return 

  def function_rank_info():
      # code
      return 

  def function_make_reccomendation():
      # code
      return 
