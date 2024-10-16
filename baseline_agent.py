from enum import Enum

class AgentState(Enum):
    IDLE = 0
    INPUT_PROCESSING = 1
    INFORMATION_GATHERING = 2
    EVIDENCE_ANALYSIS = 3
    REASONING = 4
    RECOMMENDATION_FORMULATION = 5
    OUTPUT_GENERATION = 6
    SELF_EVALUATION = 7

class Goal:
    def __init__(self, description, applicable_conditions_states, plan):
        self.description = description
        self.conditions = applicable_conditions_states # {id_prior_state : current_state} i.e. you can transit to the state when the id of teh prio state is the one specified
        self.is_active = False
        self.is_achieved = False
        self.current_plan = None
        self.plan = plan

class FakeNewsAgent:
    def __init__(self):
      self.state = AgentState.IDLE
      self.initialise_goals()
      self.generate_procedural_plan()
      # Ontology connection
      # LLM connection

    def initialise_goals(self):
        self.main_goal = "Perform fact-checking and provide return statement"
        self.subgoals = set([
            Goal("Query the Ontology to get the information", {1 : AgentState.INFORMATION_GATHERING}, []),
            Goal("Prompt the LLM to get the information", {1 : AgentState.INFORMATION_GATHERING}, []),
            Goal("Compare and analyze the evidence", {2 : AgentState.EVIDENCE_ANALYSIS}, []),
            Goal("Compute and analyze evidences", {2 : AgentState.EVIDENCE_ANALYSIS, 3 : AgentState.REASONING}, []),
            Goal("Compare and analyze the evidence", {3 : AgentState.REASONING, 5 : AgentState.RECOMMENDATION_FORMULATION}, []),
            Goal("Present results to the user", {6 : AgentState.OUTPUT_GENERATION}, [])
        ])
        # additional subgoals
        self.additional_subgoals = set()
        self.subgoals.update(self.additional_subgoals)

    ### Agent Reasoning: Goal-based Action Planning

    def get_active_goals(self):
        return [goal for goal in self.subgoals if self.state in goal.conditions and goal.is_active]

    def get_suspended_goals(self):
        return [goal for goal in self.subgoals if goal not in self.get_active_goals(self)]

    def activate_relevant_goals(self):
        for goal in self.subgoals:
            goal.is_active = goal.conditions.get(self.state-1) == self.state

    def drop_goal(self):
        for goal in self.subgoals:
            goal.is_active = not (goal.conditions.get(self.state-1) == self.state)

    def adopt_active_goals(self):
        for goal in self.get_active_goals():
            for state in goal.conditions.values():
                if state == self.state:
                    self.adopt_plan(goal)

    def pursue_goal(self, news_item):
        self.analyze_item(news_item)
        while not self.goal_achieved():
            self.select_plan()
            self.adopt_plan()
            self.evaluate_progress()
            self.adapt_approach()

    def transition_to_state(self, new_state):
        self.state = new_state
        self.activate_relevant_goals()
        self.adopt_active_goals()

    def goal_achieved(self):
        for goal in self.get_active_goals:
            if goal.is_achieved:
                self.drop_goal(goal)

    # procedutal plan is a sequence of consequtive applicable states for that goal
    def generate_procedural_plan(self):
        for goal in self.get_active_goals():
            # get the state IDs in the list associated to the goal 
            return

    def adopt_plan():
        
        return

    def drop_plan(self):
        for goal in self.get_active_goals():
            pass
            # if goal.plan == 
        return

    def analyze_item():
        # code
        return 

    def select_plan(self):
        self.current_plan = self.generate_procedural_plan()

    def evaluate_plan():
        # code 
        return 

    def evaluate_progress():
        
        return 

    def adapt_approach():
        # code
        return 

    ### Knowledge Base Reasoning

    def function_make_query_from_nl():
        # code
        return

    def function_get_info_ontology(self):
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
