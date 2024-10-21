import numpy as np
from enum import Enum
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from ontology_service import OntologyService
from llm_service import LLMService
import logging

class AgentState(Enum):
    IDLE = 0
    INPUT_PROCESSING = 1
    INFORMATION_GATHERING = 2
    EVIDENCE_ANALYSIS = 3
    REASONING = 4
    RECOMMENDATION_FORMULATION = 5
    OUTPUT_GENERATION = 6
    SELF_EVALUATION = 7
    LEARNING = 8
    ERROR = 9

@dataclass
class Plan:
    steps: List[AgentState]
    current_step: int = 0
    success_criteria: Dict[str, float] = None
    
    def next_step(self) -> Optional[AgentState]:
        if self.current_step < len(self.steps):
            state = self.steps[self.current_step]
            self.current_step += 1
            return state
        return None

    def reset(self):
        self.current_step = 0

@dataclass
class Goal:
    description: str
    conditions: Dict[AgentState, AgentState]  # {prior_state : current_state}
    plan: Plan
    is_active: bool = False
    is_dropped: bool = False
    is_achieved: bool = False
    
    def __hash__(self):
        return hash(self.description)

class FakeNewsAgent:
    def __init__(self, ontology_service: OntologyService=None, llm_service: LLMService=None):
        self.state = AgentState.IDLE
        self.ontology_service = ontology_service
        self.llm_service = llm_service
        self.logger = logging.getLogger(__name__)
        self.analysis_results = {}
        self.current_news_item = None
        self.transition_from_state = np.full(len(AgentState), False)
        self.transition_from_state[0] = True
        self.initialise_goals()

    def initialise_goals(self):
        """Initialize the main goal and subgoals with proper plans."""
        self.main_goal = "Perform fact-checking and provide return statement"
        
        standard_plan = Plan(steps=[
            AgentState.INPUT_PROCESSING,
            AgentState.INFORMATION_GATHERING,
            AgentState.EVIDENCE_ANALYSIS,
            AgentState.REASONING,
            AgentState.RECOMMENDATION_FORMULATION,
            AgentState.OUTPUT_GENERATION,
            AgentState.SELF_EVALUATION,
            AgentState.LEARNING,
            AgentState.ERROR
        ])

        self.subgoals: Set[Goal] = {
            Goal(
                description="Process and validate input",
                conditions={AgentState.IDLE : AgentState.INPUT_PROCESSING},
                plan=Plan(steps=[AgentState.INPUT_PROCESSING])
            ),
            Goal(
                description="Query the Ontology for relevant information",
                conditions={AgentState.INPUT_PROCESSING : AgentState.INFORMATION_GATHERING},
                plan=Plan(steps=[AgentState.INFORMATION_GATHERING])
            ),
            Goal(
                description="Query LLM for analysis",
                conditions={AgentState.INPUT_PROCESSING : AgentState.INFORMATION_GATHERING},
                plan=Plan(steps=[AgentState.INFORMATION_GATHERING])
            ),
            Goal(
                description="Compare and Analyze evidence",
                conditions={AgentState.INFORMATION_GATHERING : AgentState.EVIDENCE_ANALYSIS},
                plan=Plan(steps=[AgentState.EVIDENCE_ANALYSIS])
            ),
            Goal(
                description="Compute Trust and Confidence Score",
                conditions={AgentState.EVIDENCE_ANALYSIS : AgentState.REASONING},
                plan=Plan(steps=[AgentState.REASONING])
            ),
            Goal(
                description="Formulate recommendation",
                conditions={AgentState.REASONING : AgentState.RECOMMENDATION_FORMULATION},
                plan=Plan(steps=[AgentState.RECOMMENDATION_FORMULATION])
            ),
            Goal(
                description="Present result to the user",
                conditions={AgentState.RECOMMENDATION_FORMULATION : AgentState.OUTPUT_GENERATION},
                plan=Plan(steps=[AgentState.OUTPUT_GENERATION])
            ),
            # Additional subgoals
            Goal(
                description="Mantain readiness to process new queries",
                conditions = {AgentState.IDLE : AgentState.IDLE},
                plan=Plan(steps=[AgentState.IDLE])
            ),
            Goal(
                description="Accurately interpret and classify user input",
                conditions = {}, # adopted from any prior, current 
                plan=Plan(steps=[AgentState.INPUT_PROCESSING])
            ),
            Goal(
                description="Achieve a state of informed decision-making for the user regarding the factual accuracy of the query",
                conditions = {}, # adopted from any prior, current
                plan=Plan(steps=[AgentState.REASONING])
            ),
            Goal(
                description="Test if sufficient information has been gathered before proceeding to analysis",
                conditions = {}, # adopted from any prior, current
                plan=Plan(steps=[AgentState.INFORMATION_GATHERING])
            ),
            Goal(
                description="Test the reliability and completeness of the fact-check",
                conditions = {}, # adopted from any prior, current
                plan=Plan(steps=[AgentState.SELF_EVALUATION])
            ),
            Goal(
                description="Improve fact-checking strategies based on experience",
                conditions = {}, # adopted from any prior, current
                plan=Plan(steps=[AgentState.LEARNING])
            ),
            Goal(
                description="Manage and recover from errors in the fact-checking process",
                conditions = {}, # adopted from any prior, current
                plan=Plan(steps=[AgentState.ERROR])
            ),

        }

    # transitioning once the prior state conditions are satisfied 
    def transition_to_state(self, new_state: AgentState) -> None:
        """Transition to a new state and handle related goal updates."""
        if self.transition_from_state[self.state.value]:
            self.logger.info(f"Transitioning from {self.state} to {new_state}")
            self.state = new_state
            print(self.state)
            self.activate_relevant_goals()
            self.adopt_active_goals()
        else:
            self.logger.info(f"Failed Transitioning from {self.state} to {new_state}")
    
    # transition to the next state specified by the state values!
    def procedural_state_transition(self) -> None:
        """Advance to the next logical state in the processing pipeline."""
        state_order = [state for state in AgentState]
        current_index = state_order.index(self.state)
        if current_index < len(state_order) - 1:
            self.transition_to_state(state_order[current_index + 1])

    # goals with is_active set to true
    def get_active_goals(self) -> List[Goal]:
        """Return currently active goals."""
        return [goal for goal in self.subgoals if goal.is_active]
    
    # goals excluded from active goals
    def get_suspended_goals(self) -> List[Goal]:
        """Return currently suspended goals."""
        return [goal for goal in self.subgoals if not goal.is_active]

    # A goal is activated when the current state is included in the goals states 
    def activate_relevant_goals(self) -> None:
        """Activate goals relevant to the current state."""
        current_state_id = self.state.value
        for goal in self.subgoals:
            if goal.conditions:
                for prior_state in goal.conditions.keys():
                    if current_state_id - 1 == goal.conditions[prior_state]:
                        goal.is_active = goal.conditions[current_state_id - 1] == self.state
            else:
                goal.is_active = True # can be activated with no conditions

    # execute active goal(s)'s plan  
    def adopt_active_goals(self) -> None:
        """Adopt plans for active goals."""
        for goal in self.get_active_goals():
            if goal.plan:
                self.execute_plan(goal.plan)

    # the next state is determined by the current goal(s)'s plan
    def execute_plan(self, plan: Plan) -> None:
        """Execute the steps in a plan."""
        while (next_state := plan.next_step()) is not None:
            try:
                self.execute_state_action(next_state)
            except Exception as e:
                self.logger.error(f"Error executing state {next_state}: {str(e)}")
                self.transition_to_state(AgentState.SELF_EVALUATION)
                break
    
    # state-action mapping
    def execute_state_action(self, state: AgentState) -> None:
        """Execute the appropriate action for the given state."""
        action_map = {
            AgentState.INPUT_PROCESSING : self.process_input,
            AgentState.INFORMATION_GATHERING : self.gather_information,
            AgentState.EVIDENCE_ANALYSIS : self.analyze_evidence,
            AgentState.REASONING : self.perform_reasoning,
            AgentState.RECOMMENDATION_FORMULATION : self.formulate_recommendation,
            AgentState.OUTPUT_GENERATION : self.generate_output,
            AgentState.SELF_EVALUATION : self.perform_self_evaluation
        }
        
        if state in action_map.keys:
            action_map[state]()
        else:
            raise ValueError(f"No action defined for state {state}")


    ### Implementation of state-specific actions

    def process_input(self) -> None:
        """Validate and process the input news item."""
        if not self.current_news_item:
            raise ValueError("No news item to process")
        
        required_fields = ['title', 'content', 'source']
        if not all(field in self.current_news_item for field in required_fields):
            raise ValueError("Missing required fields in news item")
        
        self.analysis_results['processed_input'] = {
            'title': self.current_news_item['title'],
            'content_length': len(self.current_news_item['content']),
            'source': self.current_news_item['source']
        }
        self.transition_to_state(AgentState.INFORMATION_GATHERING)

    def gather_information(self) -> None:
        """Gather information from both ontology and LLM."""
        ontology_results = self.query_ontology()
        llm_results = self.query_llm()
        
        self.analysis_results['gathered_info'] = {
            'ontology_data': ontology_results,
            'llm_analysis': llm_results
        }

    def analyze_evidence(self) -> None:
        """Analyze gathered evidence."""
        if 'gathered_info' not in self.analysis_results:
            raise ValueError("No gathered information to analyze")
        
        self.analysis_results['evidence_analysis'] = self.rank_evidence()

    def perform_reasoning(self) -> None:
        """Perform reasoning based on analyzed evidence."""
        if 'evidence_analysis' not in self.analysis_results:
            raise ValueError("No analyzed evidence for reasoning")
        
        self.analysis_results['reasoning_results'] = self.reason_about_evidence()

    def formulate_recommendation(self) -> None:
        """Formulate a recommendation based on reasoning."""
        if 'reasoning_results' not in self.analysis_results:
            raise ValueError("No reasoning results for recommendation")
        
        self.analysis_results['recommendation'] = self.generate_recommendation()

    def generate_output(self) -> None:
        """Generate the final output."""
        if 'recommendation' not in self.analysis_results:
            raise ValueError("No recommendation to output")
        
        self.analysis_results['final_output'] = {
            'verification_result': self.analysis_results['recommendation'],
            'trust_score': self.calculate_trust_score(), 
            'confidence_score': self.calculate_confidence_score(),
            'evidence_summary': self.summarize_evidence()
        }

    def perform_self_evaluation(self) -> None:
        """Perform self-evaluation of the analysis process."""
        self.analysis_results['evaluation'] = {
            'process_complete': bool(self.analysis_results.get('final_output')),
            'confidence_level': self.calculate_confidence_score(),
            'areas_for_improvement': self.identify_improvements()
        }


    ### Helper methods

    def query_ontology(self) -> dict:
        """Query the ontology for relevant information."""
        if self.ontology_service:
            return self.ontology_service.query(self.current_news_item)
        return {}
    
    def query_llm(self) -> dict:
        """Query the LLM for analysis."""
        if self.llm_service:
            return self.llm_service.query(self.current_news_item)
        return {}

    def rank_evidence(self) -> dict:
        """Rank and score gathered evidence."""
        gathered_info = self.analysis_results['gathered_info']
        # Implement evidence ranking logic
        return {'evidence_scores': {}, 'reliability_metrics': {}}

    def reason_about_evidence(self) -> dict:
        """Perform reasoning about the evidence."""
        evidence = self.analysis_results['evidence_analysis']
        # Implement reasoning logic
        return {'conclusions': [], 'confidence_levels': {}}

    def generate_recommendation(self) -> dict:
        """Generate a recommendation based on reasoning."""
        reasoning = self.analysis_results['reasoning_results']
        # Implement recommendation generation logic
        return {'verdict': '', 'explanation': '', 'supporting_evidence': []}

    def calculate_trust_score(self) -> float:
        """Calculate the confidence score of the analysis."""
        return (self.w_ontology * self.ontology_score + self.w_llm * self.llm_score) / (self.w_ontology + self.w_llm)

    def calculate_confidence_score(self) -> float:
        """Calculate the confidence score of the analysis."""
        # Implement confidence score calculation
        return 0.0

    def summarize_evidence(self) -> dict:
        """Summarize the evidence used in the analysis."""
        # Implement evidence summarization
        return {'key_points': [], 'sources': []}

    def identify_improvements(self) -> List[str]:
        """Identify areas for improvement in the analysis process."""
        # Implement improvement identification
        return []


    ### Agent Test method

    def analyze_news_item(self, news_item: str) -> dict:
        """Main method to analyze a news item."""
        self.current_news_item = news_item
        self.analysis_results = {}
        
        try:
            self.transition_to_state(AgentState.INPUT_PROCESSING)
            
            # iterate over states and stop at the end of the cycle
            while self.state != AgentState.IDLE:
                # get active goals
                print(self.state)
                active_goals = self.get_active_goals()
                
                # re-initialise goals in case of fail
                if not active_goals:
                    # end of the cycle 
                    if self.state == AgentState.OUTPUT_GENERATION:
                        self.transition_to_state(AgentState.IDLE)
                    # automated plan rules failed, procedural state generation
                    else:
                        self.procedural_state_transition()
                
                # pursue goal
                for goal in active_goals:
                    self.execute_plan(goal.plan)
                
            return self.analysis_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing news item: {str(e)}")
            self.transition_to_state(AgentState.SELF_EVALUATION)
            raise

if __name__ == '__main__':
    FNA = FakeNewsAgent(OntologyService, LLMService)
    FNA.analyze_news_item('Does eating spicy food cause hair loss')
