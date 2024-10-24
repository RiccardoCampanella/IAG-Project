from py2pddl import Domain, create_type
from py2pddl import predicate, action, goal, init
import numpy as np
from enum import Enum
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from ontology_service import OntologyService
from llm_service import LLMService
import logging
from arguments_examples import argument_examples
from baseline_agent import AgentState, Goal, Plan, FakeNewsAgent 

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('planner.log'),
        logging.StreamHandler()
    ]
)

# Create PDDL types
State = create_type("State")
Resource = create_type("Resource")
Info = create_type("Info")

class FactCheckDomain(Domain):
    """PDDL Domain for fact-checking planning"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__ + '.FactCheckDomain')
        self.logger.info("Initializing PDDL Domain for fact-checking")
    
    @predicate(State)
    def at_state(self, state):
        """Current state predicate"""
        self.logger.debug(f"Defining state predicate for: {state}")
    
    @predicate(State, State)
    def can_transition(self, from_state, to_state):
        """Valid state transition predicate"""
        self.logger.debug(f"Defining transition predicate: {from_state} -> {to_state}")
        
    @predicate(Info)
    def has_info(self, info):
        """Information availability predicate"""
        self.logger.debug(f"Defining information predicate for: {info}")
        
    @predicate(Resource)
    def has_resource(self, resource):
        """Resource availability predicate"""
        self.logger.debug(f"Defining resource predicate for: {resource}")

    @action(State, State)
    def transition(self, from_state, to_state):
        """Action for transitioning between states"""
        self.logger.info(f"Defining transition action: {from_state} -> {to_state}")
        precond = [
            self.at_state(from_state),
            self.can_transition(from_state, to_state)
        ]
        effect = [
            ~self.at_state(from_state),
            self.at_state(to_state)
        ]
        return precond, effect
    
    @action(Info)
    def gather_info(self, info):
        """Action for gathering information"""
        self.logger.info(f"Defining gather info action for: {info}")
        precond = [
            self.at_state("INFORMATION_GATHERING"),
            ~self.has_info(info)
        ]
        effect = [
            self.has_info(info)
        ]
        return precond, effect

class ExpertReasonerAgent(FakeNewsAgent):
    def __init__(self, ontology_service: OntologyService=None, llm_service: LLMService=None):
        super().__init__(ontology_service, llm_service)
        self.logger = logging.getLogger(__name__ + '.PlannerAgent')
        self.logger.info("Initializing PlannerAgent")
        self.domain = FactCheckDomain()
        self.initialize_pddl_state()
        
    def initialize_pddl_state(self):
        """Initialize the PDDL state with predicates"""
        self.logger.info("Initializing PDDL state")
        self.pddl_state = {
            'current_state': self.state,
            'valid_transitions': {
                AgentState.IDLE: [AgentState.INPUT_PROCESSING],
                AgentState.INPUT_PROCESSING: [AgentState.INFORMATION_GATHERING],
                AgentState.INFORMATION_GATHERING: [AgentState.EVIDENCE_ANALYSIS],
                AgentState.EVIDENCE_ANALYSIS: [AgentState.REASONING, AgentState.RECOMMENDATION_FORMULATION],
                AgentState.REASONING: [AgentState.RECOMMENDATION_FORMULATION],
                AgentState.RECOMMENDATION_FORMULATION: [AgentState.SELF_EVALUATION],
                AgentState.SELF_EVALUATION: [AgentState.LEARNING, AgentState.IDLE],
                AgentState.LEARNING: [AgentState.IDLE],
                AgentState.ERROR: [AgentState.IDLE]
            },
            'resources': {
                'ontology': bool(self.ontology_service),
                'llm': bool(self.llm_service)
            },
            'info': {
                'input_processed': False,
                'evidence_gathered': False,
                'analysis_complete': False
            }
        }
        self.logger.debug(f"PDDL state initialized with resources: {self.pddl_state['resources']}")
        
    def generate_pddl_problem(self, goal_state: AgentState) -> dict:
        """Generate PDDL problem for reaching goal state"""
        self.logger.info(f"Generating PDDL problem for goal state: {goal_state}")
        
        objects = {
            'states': list(AgentState),
            'resources': list(self.pddl_state['resources'].keys()),
            'info': list(self.pddl_state['info'].keys())
        }
        self.logger.debug(f"Problem objects defined: {objects}")
        
        init_predicates = [
            self.domain.at_state(self.state.name)
        ]
        
        # Add valid transitions
        for from_state, to_states in self.pddl_state['valid_transitions'].items():
            for to_state in to_states:
                self.logger.debug(f"Adding transition predicate: {from_state} -> {to_state}")
                init_predicates.append(
                    self.domain.can_transition(from_state.name, to_state.name)
                )
        
        # Add resource predicates
        for resource, available in self.pddl_state['resources'].items():
            if available:
                self.logger.debug(f"Adding resource predicate: {resource}")
                init_predicates.append(self.domain.has_resource(resource))
        
        # Add info predicates
        for info, gathered in self.pddl_state['info'].items():
            if gathered:
                self.logger.debug(f"Adding info predicate: {info}")
                init_predicates.append(self.domain.has_info(info))
        
        goal_predicates = [
            self.domain.at_state(goal_state.name)
        ]
        
        self.logger.info(f"PDDL problem generated with {len(init_predicates)} initial predicates")
        return {
            'domain': self.domain,
            'objects': objects,
            'init': init_predicates,
            'goal': goal_predicates
        }

    def plan_path_to_goal(self, goal_state: AgentState) -> Optional[Plan]:
        """Generate plan to reach goal state using PDDL"""
        self.logger.info(f"Planning path to goal state: {goal_state}")
        try:
            problem = self.generate_pddl_problem(goal_state)
            self.logger.debug("PDDL problem generated successfully")
            
            plan_steps = self.solve_pddl_problem(problem)
            
            if plan_steps:
                self.logger.info(f"Plan generated successfully with {len(plan_steps)} steps")
                return Plan(steps=[AgentState[step] for step in plan_steps])
            else:
                self.logger.warning("No valid plan found")
                return None
            
        except Exception as e:
            self.logger.error(f"Error generating plan: {str(e)}", exc_info=True)
            return None
            
    def solve_pddl_problem(self, problem: dict) -> List[str]:
        """Solve PDDL problem using state space search"""
        self.logger.info("Starting PDDL problem solving")
        current_state = self.state
        goal_state = problem['goal']
        plan = []
        
        self.logger.debug(f"Initial state: {current_state}, Goal state: {goal_state}")
        
        while current_state != goal_state:
            # Get valid next states
            valid_next_states = self.pddl_state['valid_transitions'].get(current_state, [])
            self.logger.debug(f"Valid next states from {current_state}: {valid_next_states}")
            
            if not valid_next_states:
                self.logger.error(f"No valid transitions from {current_state}")
                return []
                
            # Choose next state using heuristic
            next_state = self.select_next_state(valid_next_states, goal_state)
            self.logger.info(f"Selected next state: {next_state}")
            
            plan.append(next_state.name)
            current_state = next_state
            
        self.logger.info(f"Plan solved with {len(plan)} steps: {plan}")
        return plan
        
    def select_next_state(self, valid_states: List[AgentState], goal_state: AgentState) -> AgentState:
        """Select next state using heuristic distance to goal"""
        self.logger.debug(f"Selecting next state from {len(valid_states)} valid states")
        selected = min(valid_states, 
                      key=lambda s: self.estimate_distance(s, goal_state))
        self.logger.debug(f"Selected state {selected} with minimum distance to goal")
        return selected
                  
    def estimate_distance(self, state: AgentState, goal_state: AgentState) -> int:
        """Estimate steps needed to reach goal state"""
        if state == goal_state:
            return 0
            
        # Count minimum transitions needed based on state order
        state_order = list(AgentState)
        current_idx = state_order.index(state)
        goal_idx = state_order.index(goal_state)
        distance = abs(goal_idx - current_idx)
        
        self.logger.debug(f"Estimated distance from {state} to {goal_state}: {distance}")
        return distance
        
    def analyze_news_item(self, news_item: str) -> dict:
        """Override analyze_news_item to use PDDL planning"""
        self.logger.info(f"Starting analysis of news item: {news_item}")
        self.current_news_item = news_item
        self.analysis_results = {}
        
        try:
            # Plan path to recommendation state
            self.logger.info("Generating plan to recommendation state")
            plan = self.plan_path_to_goal(AgentState.RECOMMENDATION_FORMULATION)
            
            if not plan:
                self.logger.error("Failed to generate valid plan")
                raise ValueError("Could not generate valid plan")
            
            self.logger.info(f"Executing plan with {len(plan.steps)} steps")
            # Execute plan steps
            for state in plan.steps:
                self.logger.debug(f"Transitioning to state: {state}")
                self.transition_to_state(state)
                self.logger.debug(f"Executing action for state: {state}")
                self.execute_state_action(state)
                
            # Return to idle
            self.logger.info("Analysis complete, returning to idle state")
            self.transition_to_state(AgentState.IDLE)
            return self.analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in analysis: {str(e)}", exc_info=True)
            self.transition_to_state(AgentState.ERROR)
            return {'error': str(e)}

if __name__ == '__main__':
    # Set up logging for main execution
    logging.info("Starting PlannerAgent execution")

    agent = ExpertReasonerAgent(OntologyService(), LLMService())
    results = agent.analyze_news_item("Does eating spicy food cause hair loss?")
    
    logging.info("Analysis complete")
    logging.debug(f"Results: {results}")