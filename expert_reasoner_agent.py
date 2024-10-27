import numpy as np
import os
from enum import Enum
from typing import Dict, List, Set, Optional, Any, Iterator, FrozenSet
from dataclasses import dataclass
from ontology_service import OntologyService
from llm_service import LLMService
import logging
from arguments_examples import argument_examples
from collections import deque
from datetime import datetime

class AgentState(Enum):
    IDLE = 0
    INPUT_PROCESSING = 1
    INFORMATION_GATHERING = 2
    EVIDENCE_ANALYSIS = 3
    RECOMMENDATION_FORMULATION = 4
    SELF_EVALUATION = 5
    REASONING = 6
    LEARNING = 7
    ERROR = 8

@dataclass
class Plan:
    steps: List[AgentState]
    current_step: int = 0
    success_criteria: Dict[str, float] = None
    is_suspended = False
    
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
    is_achievable: bool = False
    is_suspended: bool = False
    is_replanned: bool = False

    
    def __hash__(self):
        return hash(self.description)
    
    def __iter__(self) -> Iterator[Any]:
        yield self.description
        yield from self.conditions.items()  # Yields (key, value) pairs from conditions
        yield self.plan 
        yield self.is_active
        yield self.is_dropped
        yield self.is_achieved
        yield self.is_achievable
        yield self.is_suspended
        yield self.is_replanned

class FakeNewsAgent:

    def __init__(self, ontology_service: OntologyService=None, llm_service: LLMService=None):
        self.ontology_service = ontology_service
        self.llm_service = llm_service
        self.analysis_results = {} 
        self.current_news_item = "Eating spicy food causes hair loss" #"Running is good for your health" 
        self.state = AgentState.IDLE
        self.initialise_goals()
        self.hyperparameters = self.initialise_hyperparameters()
        self.agent_memory = self.subgoals.copy()
        self.logger = self.setup_logger()

    def initialise_goals(self):
        """Initialize the main goal and subgoals with proper plans."""
        self.main_goal = "Perform fact-checking and provide return statement"
        
        # State-Flow of a standard plan 
        standard_plan = Plan(steps=[
            AgentState.INPUT_PROCESSING,
            AgentState.INFORMATION_GATHERING,
            AgentState.EVIDENCE_ANALYSIS,
            AgentState.REASONING,
            AgentState.RECOMMENDATION_FORMULATION,
            AgentState.SELF_EVALUATION,
            AgentState.LEARNING,
            AgentState.ERROR
        ])

        # Main goal's Plan is broken down into subgoals with a subplan each 
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
                description="Formulate recommendation",
                conditions={AgentState.EVIDENCE_ANALYSIS : AgentState.RECOMMENDATION_FORMULATION},
                plan=Plan(steps=[AgentState.RECOMMENDATION_FORMULATION])
            ),
            # Additional subgoals
            Goal(
                description="Leran from Slef Evaluation",
                conditions={AgentState.RECOMMENDATION_FORMULATION : AgentState.SELF_EVALUATION},
                plan=Plan(steps=[AgentState.SELF_EVALUATION])
            ),
            Goal(
                description="Return to the Initial State",
                conditions={AgentState.SELF_EVALUATION : AgentState.IDLE},
                plan=Plan(steps=[AgentState.IDLE])
            ),
            Goal(
                description="Await for user input",
                conditions={AgentState.IDLE : AgentState.IDLE},
                plan=Plan(steps=[AgentState.IDLE])
            )
            }
            # Additional subgoals that can be integrated 
        """
        Goal(
            description="Compute Trust and Confidence Score",
            conditions={AgentState.EVIDENCE_ANALYSIS : AgentState.REASONING},
            plan=Plan(steps=[AgentState.REASONING])
            ),
        Goal(
            description="Accurately interpret and classify user input",
            conditions = {}, # include all the states if the goal is adopted from any prior, current 
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
        ),"""
       
    # transitioning once the prior state conditions are satisfied 
    def transition_to_state(self, new_state: AgentState) -> None:
        """Transition to a new state and handle related goal updates."""
        self.logger.info(f"Transitioning from {self.state} to {new_state}")
        self.agent_memory.add(frozenset(self.subgoals.copy())) # update agent memory
        self.deactivate_goals() #TODO check assumption 
        self.state = new_state
        self.activate_relevant_goals()
        self.agent_memory.add(frozenset(self.subgoals.copy()))
        if not self.get_active_goals():
            self.logger.info(f"Failed Transitioning from {self.state} to {new_state}")
    
    # goals with is_active set to true
    def get_active_goals(self) -> List[Goal]:
        return [goal for goal in self.subgoals if goal.is_active]
    
    def deactivate_goals(self):
        for goal in self.subgoals:
            goal.is_active = False
            # a goal is dropped the state in its plane list is reached => still managed to carry out task (agent successfull)
            if goal.plan.steps:
                if goal.plan.steps[-1] == self.state: # assume the plan list is ordered in sequential logical order
                    goal.is_dropped = True
                    goal.is_achieved = True
            elif goal.is_achievable:
                 # if a goal is not reached it can be resumed later (unless unachievable or irrelevant)
                 goal.is_suspended = True
                 goal.plan.is_suspended = True
                 goal.plan.current_step = goal.plan.steps.index(self.state)
            else:
                 goal.is_suspended = False
                 goal.is_achievable = False
                 goal.is_dropped = True
                 goal.plan.steps = [] # Dastani paper's rule 
        self.logger.info(f"\n Deactivating prior goals: {[goal.description for goal in self.subgoals]}, \n achieved: {[goal.description for goal in self.subgoals if goal.is_achieved]}, \n suspended: {[goal.description for goal in self.subgoals if goal.is_suspended]}")


    # goals excluded from active goals
    def get_suspended_goals(self) -> List[Goal]:
        """Return currently suspended goals."""
        suspended_goals =  [goal for goal in self.subgoals if not goal.is_active and goal.is_suspended]
        self.logger.info(f"get_suspended_goals: {[goal.description for goal in suspended_goals]}")
        return suspended_goals

    # A goal is activated when the current state is included in the goals states 
    def activate_relevant_goals(self) -> None:
        """Activate goals relevant to the current state."""
        current_state_id = self.state.value
        for goal in self.subgoals:
            if goal.conditions:
                if not goal.is_suspended and not goal.plan:
                    self.logger.info(f"Failed to activate_relevant_goals")
                else:
                    for prior_state in goal.conditions.keys():
                        if current_state_id - 1 == prior_state.value:
                            goal.is_active = True
            else:
                goal.is_active = True # can be activated with no conditions
        self.logger.info(f"activate_relevant_goals: {[goal.description for goal in self.get_active_goals()]}")
       
    # execute active goal(s)'s plan  
    def adopt_active_goals(self) -> None:
        """Adopt plans for active goals."""
        active_goals = self.get_active_goals()
        self.logger.info(f"Adopt_active_goals: {[goal.description for goal in active_goals]}")
        for goal in active_goals:
            if goal.plan.steps:
                self.execute_plan(goal.plan)
                self.deactivate_goals()
            else:
                goal.is_achieved = True
                
    # the next state is determined by the current goal(s)'s plan
    def execute_plan(self, plan: Plan) -> None:
        """Execute the steps in a plan."""
        self.logger.info(f"execute_plan, {self.state}")
        while (next_state := plan.next_step()) is not None:
            try:
                self.execute_state_action(next_state)
            except Exception as e:
                self.logger.error(f"Error executing state {next_state}: {str(e)}")
                #self.transition_to_state(AgentState.SELF_EVALUATION)
                break
        
    # state-action mapping
    def execute_state_action(self, state: AgentState) -> None:
        """Execute the appropriate action for the given state."""
        action_map = {
            AgentState.IDLE : self.await_user,
            AgentState.INPUT_PROCESSING : self.process_input,
            AgentState.INFORMATION_GATHERING : self.gather_information,
            AgentState.EVIDENCE_ANALYSIS : self.analyze_evidence,
            #AgentState.REASONING : self.knowledge_reasoning_match,
            AgentState.RECOMMENDATION_FORMULATION : self.formulate_recommendation,
            AgentState.SELF_EVALUATION : self.perform_self_evaluation
        }
        
        if state in action_map.keys():
            action_map[state]()
            self.logger.info(f"execute_state_action, state: {self.state}, action: {action_map[state]}, goals active: {[goal.description for goal in self.get_active_goals()]}")
        else:
            raise ValueError(f"No action defined for state {state}")
    
    # the agent will tune this knob to imporve itself
    def initialise_hyperparameters(self):
        hyperparameters = {
            'trust_ontology' : 0.8, # assigned by us
            'trust_llm' : 0.8, # assigned by us, will be multiplied by the trust the llm has in its answer.
            'trust_llm_vedant' : 0.7 
        }
        return hyperparameters





    ################################ Logic for Planning the Agent Next State ################################
    def identify_next_state(self) -> AgentState:
        """
        Identify the next state based on goal conditions, plan steps, and state transitions.
        Uses a graph-based approach to find the optimal path to the final state while
        ensuring necessary goals are completed.
        
        Returns:
            AgentState: The next state the agent should transition to
        """
        self.logger.debug(f"Identifying next state from current state: {self.state}")
        
        # Build state transition graph from goals and conditions
        state_graph = self._build_state_graph()
        
        # Get candidate next states based on current state and goal conditions
        candidate_states = self._get_candidate_states()
        
        if not candidate_states:
            self.logger.warning("No candidate states found, using procedural fallback")
            return self._get_procedural_next_state()
        
        # Find paths to final state for each candidate
        paths_to_final = {}
        final_states = {AgentState.IDLE}
        
        for candidate in candidate_states:
            shortest_path = self._find_shortest_path(
                state_graph,
                candidate,
                final_states
            )
            if shortest_path:
                paths_to_final[candidate] = shortest_path
        
        # Select optimal next state based on path analysis
        next_state = self._select_optimal_state(paths_to_final)
        
        self.logger.info(f"Selected next state: {next_state}")
        return next_state

    def _build_state_graph(self) -> Dict[AgentState, Set[AgentState]]:
        """
        Build a graph of state transitions from goal conditions and plans.
        
        Returns:
            Dict[AgentState, Set[AgentState]]: Graph representing possible state transitions
        """
        state_graph = {state: set() for state in AgentState}
        
        # Add transitions from goal conditions
        for goal in self.subgoals:
            if goal.conditions:
                for prior_state, current_state in goal.conditions.items():
                    state_graph[prior_state].add(current_state)
            
            # Add transitions from plan steps
            if goal.plan and goal.plan.steps:
                for i in range(len(goal.plan.steps) - 1):
                    current_step = goal.plan.steps[i]
                    next_step = goal.plan.steps[i + 1]
                    state_graph[current_step].add(next_step)
        
        return state_graph

    def _get_candidate_states(self) -> Set[AgentState]:
        """
        Get valid candidate states based on current state and goal conditions.
        
        Returns:
            Set[AgentState]: Set of possible next states
        """
        candidates = set()
        current_state_id = self.state.value
        
        # Add states from goal conditions
        for goal in self.subgoals:
            if goal.conditions:
                for prior_state, curr_state in goal.conditions.items():
                    if prior_state == self.state:
                        candidates.add(curr_state)
            
            # Add states from plan steps
            if goal.plan and goal.plan.steps:
                try:
                    current_idx = goal.plan.steps.index(self.state)
                    if current_idx < len(goal.plan.steps) - 1:
                        candidates.add(goal.plan.steps[current_idx + 1])
                except ValueError:
                    continue
        
        return candidates

    def _find_shortest_path(
        self,
        graph: Dict[AgentState, Set[AgentState]],
        start: AgentState,
        end_states: Set[AgentState]
    ) -> Optional[List[AgentState]]:
        """
        Find shortest path from start state to any of the end states using BFS.
        
        Args:
            graph: State transition graph
            start: Starting state
            end_states: Set of possible end states
        
        Returns:
            Optional[List[AgentState]]: Shortest path if found, None otherwise
        """
        if start in end_states:
            return [start]
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current_state, path = queue.popleft()
            
            for next_state in graph[current_state]:
                if next_state in end_states:
                    return path + [next_state]
                
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, path + [next_state]))
        
        return None

    def _select_optimal_state(
        self,
        paths: Dict[AgentState, List[AgentState]]
    ) -> AgentState:
        """
        Select the optimal next state based on path analysis and goal completion.
        
        Args:
            paths: Dictionary mapping candidate states to their paths to final state
        
        Returns:
            AgentState: Selected next state, or procedural fallback if no valid paths
        """
        if not paths:
            return self._get_procedural_next_state()
        
        # Score each path based on multiple criteria
        path_scores = {}
        for state, path in paths.items():
            score = self._calculate_path_score(state, path)
            path_scores[state] = score
        
        # Select state with highest score
        return max(path_scores.items(), key=lambda x: x[1])[0]

    def _calculate_path_score(
        self,
        state: AgentState,
        path: List[AgentState]
    ) -> float:
        """
        Calculate path score incorporating learning metrics, historical performance,
        and self-improvement indicators.
        
        Args:
            state: Candidate next state
            path: Path from state to final state
        
        Returns:
            float: Comprehensive score for the path
        """
        # Base path efficiency score (inverse of path length)
        base_score = 1.0 / len(path)
        
        # Initialize scoring components
        completion_score = 0.0
        learning_score = 0.0
        efficiency_score = 0.0
        confidence_score = 0.0
        
        # Calculate goal completion score
        goals_completed = sum(
            1 for goal in self.subgoals
            if all(s in path for s in goal.plan.steps)
        )
        completion_score = 0.2 * (goals_completed / len(self.subgoals))
        
        # Calculate learning-based improvements score
        learning_score = self._calculate_learning_score(state, path)
        
        # Calculate efficiency score based on historical performance
        efficiency_score = self._calculate_efficiency_score(state, path)
        
        # Calculate confidence score based on past evaluations
        confidence_score = self._calculate_confidence_score(state, path)
        
        # Apply penalties
        penalties = self._calculate_penalties(state, path)
        
        # Combine scores with weighted importance
        final_score = (
            base_score * 0.15 +          # Base path efficiency
            completion_score * 0.20 +     # Goal completion
            learning_score * 0.25 +       # Learning improvements (now missing REASONING component)
            efficiency_score * 0.20 +     # Historical efficiency
            confidence_score * 0.20 -     # Confidence in decisions
            penalties                     # Various penalties
        )
        
        self.logger.debug(f"""Path score components for state {state}:
            Base: {base_score:.3f}
            Completion: {completion_score:.3f}
            Learning: {learning_score:.3f}
            Efficiency: {efficiency_score:.3f}
            Confidence: {confidence_score:.3f}
            Penalties: {penalties:.3f}
            Final: {final_score:.3f}""")
        
        return final_score

    def _calculate_learning_score(
        self,
        state: AgentState,
        path: List[AgentState]
    ) -> float:
        """
        Calculate learning-based improvement score from historical performance
        and hyperparameter adjustments.
        """
        learning_score = 0.0
        
        # Check if we have learning history in analysis_results
        if hasattr(self, 'analysis_results') and 'learning' in self.analysis_results:
            learning_history = self.analysis_results['learning']
            
            # Score based on hyperparameter optimization
            if 'hyperparameter_adjustments' in learning_history:
                adjustments = learning_history['hyperparameter_adjustments']
                
                # Reward paths that utilize highly trusted components
                if state in {AgentState.INFORMATION_GATHERING, AgentState.EVIDENCE_ANALYSIS}:
                    learning_score += adjustments.get('trust_ontology', 0.5) * 0.3
                    learning_score += adjustments.get('trust_llm', 0.5) * 0.3
            
            # Score based on performance metrics improvement
            if 'performance_metrics' in learning_history:
                metrics = learning_history['performance_metrics']
                learning_score += metrics.get('achieved_goals_ratio', 0.5) * 0.2
                learning_score += (1 - metrics.get('dropped_goals_ratio', 0.5)) * 0.2
        
        # Consider self-evaluation states in path
        eval_states = {AgentState.SELF_EVALUATION}
        if any(s in path for s in eval_states):
            learning_score *= 1.2  # Bonus for paths including learning opportunities
        
        return learning_score

    def _calculate_efficiency_score(
        self,
        state: AgentState,
        path: List[AgentState]
    ) -> float:
        """
        Calculate efficiency score based on historical state transitions
        and execution patterns.
        """
        efficiency_score = 0.0
        
        # Analyze agent memory for successful patterns
        if hasattr(self, 'agent_memory'):
            memory_states = list(self.agent_memory)
            if memory_states:
                # Check if this path matches previously successful patterns
                successful_patterns = self._identify_successful_patterns(memory_states)
                path_pattern = tuple(path)
                if path_pattern in successful_patterns:
                    efficiency_score += 0.3
                
                # Reward paths that avoid historically problematic state sequences
                problematic_patterns = self._identify_problematic_patterns(memory_states)
                if not any(pattern in path_pattern for pattern in problematic_patterns):
                    efficiency_score += 0.2
        
        # Consider state transition efficiency
        transition_efficiency = self._calculate_transition_efficiency(state, path)
        efficiency_score += transition_efficiency * 0.5
        
        return efficiency_score

    def _calculate_confidence_score(
        self,
        state: AgentState,
        path: List[AgentState]
    ) -> float:
        """
        Calculate confidence score based on historical evaluations
        and success rates.
        """
        confidence_score = 0.0
        
        # Consider evaluation history if available
        if hasattr(self, 'analysis_results') and 'evaluation' in self.analysis_results:
            evaluation = self.analysis_results['evaluation']
            
            # Base confidence on historical success
            if 'confidence_level' in evaluation:
                confidence_score += evaluation['confidence_level'] * 0.4
            
            # Consider process completion rate
            if 'process_complete' in evaluation:
                confidence_score += float(evaluation['process_complete']) * 0.3
            
            # Analyze improvement areas
            if 'areas_for_improvement' in evaluation:
                improvements = evaluation['areas_for_improvement']
                # Higher confidence if fewer improvements needed
                confidence_score += (1 - (len(improvements) / 10)) * 0.3
        
        return confidence_score

    def _calculate_penalties(
        self,
        state: AgentState,
        path: List[AgentState]
    ) -> float:
        """
        Calculate penalties based on various risk factors and constraints.
        """
        penalties = 0.0
        
        # Penalty for suspended goals
        suspended_goals = [
            goal for goal in self.subgoals
            if goal.is_suspended and state in goal.plan.steps
        ]
        penalties += len(suspended_goals) * 0.1
        
        # Penalty for deviating from natural progression
        if state.value != self.state.value + 1:
            penalties += 0.1
        
        # Penalty for repeated states in path
        state_counts = {}
        for s in path:
            state_counts[s] = state_counts.get(s, 0) + 1
            if state_counts[s] > 1:
                penalties += 0.15
        
        # Penalty for skipping critical states
        critical_states = {
            AgentState.EVIDENCE_ANALYSIS,
            AgentState.RECOMMENDATION_FORMULATION,
            AgentState.SELF_EVALUATION
        }
        if any(s not in path for s in critical_states):
            penalties += 0.2
        
        return penalties

    def _identify_successful_patterns(
        self,
        memory_states: List[frozenset[Goal]]
    ) -> Set[tuple]:
        """
        Identify historically successful state transition patterns.
        """
        successful_patterns = set()
        for memory_state in memory_states:
            achieved_goals = [
                goal for goal in memory_state
                if (isinstance(goal, Goal) and goal.is_achieved) and (isinstance(goal, Goal) and not goal.is_dropped)
            ]
            for goal in achieved_goals:
                if goal.plan and goal.plan.steps:
                    successful_patterns.add(tuple(goal.plan.steps))
        return successful_patterns

    def _identify_problematic_patterns(
        self,
        memory_states: List[frozenset[Goal]]
    ) -> Set[tuple]:
        """
        Identify historically problematic state transition patterns.
        """
        problematic_patterns = set()
        for memory_state in memory_states:
            problematic_goals = [
                goal for goal in memory_state
                if (isinstance(goal, Goal) and goal.is_dropped) or (isinstance(goal, Goal) and goal.is_suspended)
            ]
            for goal in problematic_goals:
                if goal.plan and goal.plan.steps:
                    problematic_patterns.add(tuple(goal.plan.steps))
        return problematic_patterns

    def _calculate_transition_efficiency(
        self,
        state: AgentState,
        path: List[AgentState]
    ) -> float:
        """
        Calculate the efficiency of state transitions in the path.
        """
        if not path:
            return 0.0
        
        # Calculate transition costs
        transition_costs = 0
        for i in range(len(path) - 1):
            current_state = path[i]
            next_state = path[i + 1]
            
            # Higher cost for non-adjacent state transitions
            if abs(next_state.value - current_state.value) > 1:
                transition_costs += 0.2
            
            # Higher cost for backwards transitions
            if next_state.value < current_state.value:
                transition_costs += 0.3
        
        # Convert costs to efficiency score (inverse relationship)
        return 1.0 / (1.0 + transition_costs)

    def _get_procedural_next_state(self) -> AgentState:
        """
        Get next state based on procedural order as fallback.
        
        Returns:
            AgentState: Next state in procedural order
        """
        current_idx = list(AgentState).index(self.state)
        if current_idx < len(AgentState) - 1:
            return list(AgentState)[current_idx + 1]
        return AgentState.IDLE                 


    # transition to the next state specified by the state values!
    def procedural_state_transition(self) -> None:
        """Advance to the next logical state in the processing pipeline."""
        state_order = [state for state in AgentState]
        current_index = state_order.index(self.state)
        self.logger.info(f"following procedural_state_transition: State {current_index} to {current_index+1}")
        while current_index < len(state_order) - 1 and current_index!=0:
            self.transition_to_state(state_order[current_index + 1]) # assume the plan list is ordered in sequential logical order
        else:
            self.logger.info(f"Completed procedural_state_transition: State {current_index}")


    


    ################################ Implementation of State-specific Actions ################################

    def await_user(self) -> None: 
        # Display a list of sample news options for the user to prompt
        sample_news = [
            "Eating spicy food causes hair loss",
            "Running is good for your health",
            "Consuming sugary food helps with Diabetes Mellitus",
            "Meditation improves mental health",
            "Daily coffee boosts productivity",
            "Skipping breakfast helps with weight loss",
            "Smoking directly causes lung cancer in humans",
            "Eating late at night disrupts morning metabolism",
            "All humans need water to survive.",
            "Ninety percent of diets fail within one year.",
            "Vaccines prevent the spread of infectious diseases."
            "Machines are becoming sentients."
        ]
        
        print("Please choose a news article or enter your own:")
        for i, news in enumerate(sample_news, 1):
            print(f"{i}. {news}")
        
        # Allow the user to select a predefined headline or enter their own
        choice = input("Enter the number of your choice or type a new headline: ")
        
        # Check if the input is a digit and corresponds to one of the options
        if choice.isdigit() and 1 <= int(choice) <= len(sample_news):
            self.current_news_item = sample_news[int(choice) - 1]
        else:
            # Use the user's custom input
            self.current_news_item = choice
        
        print(f"Selected news article: {self.current_news_item}")

    def process_input(self) -> None:
        """Validate and process the input news item."""
        self.logger.debug(f"process_input, state : {self.state}")
        
        if not self.current_news_item:
            raise ValueError("No news item to process")
        
        # Basic validation checks
        if not isinstance(self.current_news_item, str):
            raise ValueError("News item must be a string")
        
        # Remove extra whitespace and check if empty
        cleaned_text = self.current_news_item.strip()
        if not cleaned_text:
            raise ValueError("News item cannot be empty")
        
        try:
            # 1. Check minimum and maximum word length
            words = cleaned_text.split()
            if len(words) < 3:
                raise ValueError("Input is too short (minimum 3 words)")
            if len(words) > 50:
                raise ValueError("Input is too long (maximum 50 words)")
            
            # 2. Check if it ends with proper punctuation
            if not cleaned_text[-1] in ['.', '!', '?']:
                raise ValueError("Sentence must end with proper punctuation (., !, or ?)")
            
            # If all checks pass, update the current news item with cleaned version
            self.current_news_item = cleaned_text
            self.logger.info(f"Valid input processed: {self.current_news_item}")
            
        except Exception as e:
            self.logger.error(f"Error processing input: {str(e)}")
            raise ValueError(f"Invalid input: {str(e)}")

    def gather_information(self) -> None:
        """Gather information from both ontology and LLM."""
        self.logger.debug(f"gather_information, state : {self.state}")
        print("Gathering information")
        ontology_results = []

        if self.ontology_service:
            query_arguments, target, forall = self.ontology_service.nlp_to_protege_query(self.current_news_item)
            print(F"RESULT ONTOLOGY : {query_arguments, target, forall}") 
            ontology_results = self.ontology_service.query(query_arguments, target, forall)
        
        llm_results = []
        trust = 1
        if self.llm_service != None:
            try:
                llm_results, trust = self.llm_service.query(self.current_news_item)
            except Exception as e :
                print(e)
                # here we could call a backup llm in the advanced agent model
        
        self.analysis_results['gathered_info'] = {
            "ontology_data": ontology_results,
            "llm_analysis": {
                'llm_args':llm_results,
                'llm_selftrust':trust
            }
        }

    def analyze_evidence(self) -> None:
        """Analyze gathered evidence."""
        self.logger.debug(f"current_function: analyze_evidence, state : {self.state}")
        if 'gathered_info' not in self.analysis_results:
            raise ValueError("No gathered information to analyze")
        
        self.confidence = 0.5 # Set base confidence. 0.5 is neutral
        if self.analysis_results['gathered_info']['llm_analysis']['llm_args'] != []: # if there are LLM results
            llm_selftrust = self.analysis_results['gathered_info']['llm_analysis']['llm_selftrust']
            trust_llm = self.hyperparameters['trust_llm'] * abs((llm_selftrust - 0.5)* 2) # Multiply trust in LLm by trust the llm has in itself
            self.calculate_confidence_score(trust_llm, llm_selftrust < 0.5) # Calculate confidence
        if self.analysis_results['gathered_info']['ontology_data'] != None: # If there are ontology results
            self.calculate_confidence_score(self.hyperparameters['trust_ontology'], True)

        self.analysis_results['reasoning_results'] = {
            "isTrue" : self.confidence > 0.5, # if the confidence in the statement is larger than 0.5. Then the statement is true
            "confidence_percentage" : abs((self.confidence - 0.5)* 2 * 100) # Turn confidence in percentage
        }
        print(F"ANALYZE RESULT {self.analysis_results['reasoning_results']}")
        
    def formulate_recommendation(self) -> None:
        """Formulate a recommendation based on reasoning."""
        self.logger.debug(f"formulate_recomendation, state: {self.state}, goals active ")
        if 'reasoning_results' not in self.analysis_results:

            raise ValueError("No reasoning results for recommendation")
        print(self.analysis_results['reasoning_results'])

        self.analysis_results['recommendation'] = f"The statement {self.current_news_item} is determined to be "+ \
            f"{self.analysis_results['reasoning_results']['isTrue']} with a confidence margin of " + \
            f"{self.analysis_results['reasoning_results']['confidence_percentage']}%"
        
        self.analysis_results['final_output'] = {
            'verification_result': self.analysis_results['recommendation'],
            'evidence_summary_for': [arg["text"] for arg in self.analysis_results["gathered_info"]["llm_analysis"]["llm_args"] if not arg["boolCounterArgument"]],
            'evidence_summary_against': [arg["text"] for arg in self.analysis_results["gathered_info"]["llm_analysis"]["llm_args"] if arg["boolCounterArgument"]]
        }
        print(self.analysis_results["final_output"])
        
    def perform_self_evaluation(self) -> None:
        """Perform self-evaluation of the analysis process."""
        self.logger.debug(f"perfrom_self_evaluation, state : {self.state}, goals active ")
        self.analysis_results['evaluation'] = {
            'process_complete': bool(self.analysis_results.get('final_output')),
            'areas_for_improvement': self.identify_improvements()
        }
        self.learn_from_experience()

    def mantain_readyness(self):
        pass

    def learn_from_experience(self) -> None:
        """
        Adjust hyperparameters based on self-evaluation results and performance metrics.
        Called after perform_self_evaluation to tune the agent's behavior.
        """
        if not self.analysis_results.get('evaluation'):
            self.logger.warning("No evaluation results available for learning")
            return

        self.logger.info("Starting learning process to adjust hyperparameters")
        
        # Extract evaluation metrics
        evaluation = self.analysis_results['evaluation']
        confidence_level = evaluation.get('confidence_level', 0)
        process_complete = evaluation.get('process_complete', False)
        
        # Calculate performance metrics
        dropped_goals_ratio = len([g for g in self.subgoals if g.is_dropped]) / len(self.subgoals)
        suspended_goals_ratio = len(self.get_suspended_goals()) / len(self.subgoals)
        achieved_goals_ratio = len([g for g in self.subgoals if g.is_achieved]) / len(self.subgoals)
        
        # Analyze information source reliability
        llm_ontology_agreement = self._calculate_source_agreement()
        
        # Adjustment factors based on performance
        adjustment_factors = {
            'trust_ontology': self._calculate_ontology_adjustment(
                achieved_goals_ratio,
                suspended_goals_ratio,
                llm_ontology_agreement
            ),
            'trust_llm': self._calculate_llm_adjustment(
                achieved_goals_ratio,
                confidence_level,
                llm_ontology_agreement
            )
        }
        
        # Apply adjustments with learning rate and bounds
        learning_rate = 0.1
        for param, adjustment in adjustment_factors.items():
            current_value = self.hyperparameters.get(param, 0.5)
            new_value = current_value + (adjustment * learning_rate)
            # Ensure values stay within [0.1, 0.9] range
            new_value = max(0.1, min(0.9, new_value))
            self.hyperparameters[param] = new_value
            
            self.logger.info(f"Adjusted {param}: {current_value:.3f} -> {new_value:.3f}")

        # Store learning results for future reference
        self.analysis_results['learning'] = {
            'hyperparameter_adjustments': {
                param: self.hyperparameters[param] for param in adjustment_factors.keys()
            },
            'performance_metrics': {
                'achieved_goals_ratio': achieved_goals_ratio,
                'dropped_goals_ratio': dropped_goals_ratio,
                'suspended_goals_ratio': suspended_goals_ratio,
                'confidence_level': confidence_level,
                'llm_ontology_agreement': llm_ontology_agreement
            }
        }
   




    ################################ Helper methods ################################

    # confidence measure is affected by the mismatches between llm and ontology  
    def calculate_confidence_score(self, trust, boolCounterArg) -> float:
        """Calculate the confidence score of the analysis."""
        confidence = self.confidence
        print(confidence)
        if boolCounterArg:
            confidence = confidence * (1-trust)
        else: 
            confidence = confidence * (1/(1-trust))
        print(confidence)
        if confidence > 1:
            confidence = 0.5 - (self.confidence - 0.5)
            
            print(confidence)
            if not boolCounterArg:
                confidence = confidence * (1-trust)
            else: 
                confidence = confidence * (1/(1-trust))
            print(confidence)
            confidence = 1 - confidence 
            print(confidence)

        self.confidence = confidence
        return self.confidence

    def identify_improvements(self) -> List[str]:
        """
        Identify areas for improvement in the analysis process based on agent memory,
        goals state, and execution history.
        
        Returns:
            List[str]: List of identified areas for improvement
        """
        improvements = []
        
        # Check for unachievable goals
        dropped_goals = [goal for goal in self.subgoals if goal.is_dropped and not goal.is_achieved]
        if dropped_goals:
            improvements.append(
                f"Unachievable goals detected: {', '.join(goal.description for goal in dropped_goals)}. "
                "Consider adjusting goal conditions or required resources."
            )
        
        # Check for suspended goals
        suspended_goals = self.get_suspended_goals()
        if suspended_goals:
            improvements.append(
                f"Suspended goals found: {', '.join(goal.description for goal in suspended_goals)}. "
                "Review prerequisites and dependencies."
            )
        
        # Analyze transitions in agent memory for repetitive states
        if len(self.agent_memory) > 1:
            state_sequence = [goal for memory_state in self.agent_memory 
                            for goal in memory_state if isinstance(goal, Goal) and goal.is_active]
            state_counts = {}
            for goal in state_sequence:
                if goal.plan.steps:
                    current_state = goal.plan.steps[goal.plan.current_step - 1] if goal.plan.current_step > 0 else goal.plan.steps[0]
                    state_counts[current_state] = state_counts.get(current_state, 0) + 1
                    
            # Identify repeated states
            repeated_states = {state: count for state, count in state_counts.items() 
                            if count > 1}
            if repeated_states:
                improvements.append(
                    f"Detected repeated states: {', '.join(f'{state.name}({count} times)' for state, count in repeated_states.items())}. "
                    "Review state transition logic."
                )
        
        # Check LLM and Ontology service usage
        if hasattr(self, 'llm_service') and self.llm_service:
            if self.hyperparameters.get('trust_llm', 0) < 0.7:
                improvements.append(
                    f"Low LLM trust score ({self.hyperparameters.get('trust_llm')}). "
                    "Consider improving prompt engineering or model selection."
                )
        
        if hasattr(self, 'ontology_service') and self.ontology_service:
            if self.hyperparameters.get('trust_ontology', 0) < 0.8:
                improvements.append(
                    "Low ontology trust score. Review ontology query patterns "
                    "and knowledge base completeness."
                )
        
        # Check for replanned goals
        replanned_goals = [goal for goal in self.subgoals if goal.is_replanned]
        if replanned_goals:
            improvements.append(
                f"Goals requiring replanning: {', '.join(goal.description for goal in replanned_goals)}. "
                "Review initial planning strategy."
            )
        
        # Check analysis results completeness
        if hasattr(self, 'analysis_results'):
            missing_steps = []
            expected_keys = {
                'processed_input', 'gathered_info', 'evidence_analysis',
                'reasoning_results', 'final_output', 'evaluation'
            }
            missing_keys = expected_keys - set(self.analysis_results.keys())
            if missing_keys:
                improvements.append(
                    f"Incomplete analysis steps: {', '.join(missing_keys)}. "
                    "Review process completion criteria."
                )
        
        # If no improvements identified, suggest general enhancement
        if not improvements:
            improvements.append(
                "No critical issues found. Consider enhancing knowledge base "
                "and refining confidence scoring algorithms."
            )
        
        return improvements

    def _calculate_source_agreement(self) -> float:
        """
        Calculate agreement level between LLM and Ontology sources.
        Returns a value between 0 and 1.
        """
        if 'gathered_info' not in self.analysis_results:
            return 0.5
        
        gathered_info = self.analysis_results['gathered_info']
        ontology_data = gathered_info.get('ontology_data', {})
        llm_analysis = gathered_info.get('llm_analysis', {})
        
        if not ontology_data or not llm_analysis:
            return 0.5
        
        # Compare key findings between sources
        # This is a simplified comparison - extend based on your specific data structure
        try:
            agreement_score = sum(
                1 for k, v in ontology_data.items()
                if k in llm_analysis and llm_analysis[k] == v
            ) / len(ontology_data)
            return agreement_score
        except (AttributeError, ZeroDivisionError):
            return 0.5

    def _calculate_ontology_adjustment(
        self,
        achieved_ratio: float,
        suspended_ratio: float,
        source_agreement: float
    ) -> float:
        """Calculate adjustment for ontology trust based on performance metrics."""
        # Positive factors increase trust
        positive_factors = [
            achieved_ratio * 0.4,  # Weight achievement heavily
            source_agreement * 0.3  # Consider agreement with LLM
        ]
        
        # Negative factors decrease trust
        negative_factors = [
            suspended_ratio * 0.3  # Penalize suspended goals
        ]
        
        return sum(positive_factors) - sum(negative_factors)

    def _calculate_llm_adjustment(
        self,
        achieved_ratio: float,
        confidence: float,
        source_agreement: float
    ) -> float:
        """Calculate adjustment for LLM trust based on performance metrics."""
        # Positive factors increase trust
        positive_factors = [
            achieved_ratio * 0.3,
            confidence * 0.3,
            source_agreement * 0.2
        ]
        
        # Negative factors decrease trust
        negative_factors = [
            (1 - confidence) * 0.2  # Penalize low confidence
        ]
        
        return sum(positive_factors) - sum(negative_factors)
    
    def setup_logger(self):
        """
        Configure logging to write to a file with date-based naming.
        Suppresses terminal output while maintaining detailed logging in files.
        """
        # Create logs directory if it doesn't exist
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a date-based log filename
        current_date = datetime.now().strftime('%Y-%m-%d')
        log_filename = os.path.join(log_dir, f'baseline_agent_{current_date}.log')
        
        # Create a logger instance
        logger = logging.getLogger('FakeNewsTrainer')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Clear any existing handlers
        logger.handlers = []
        
        # Add the file handler to the logger
        logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        return logger




    ################################ Agent Test method ################################

    def analyze_news_item(self) -> dict:
        """Main method to analyze a news item."""
        self.analysis_results = {}

        try:
            while (next_state:=self.identify_next_state()) != AgentState.IDLE:
                # get active goals
                active_goals = self.get_active_goals()
                
                # if in self evaluation, transition to IDLE
                if not active_goals and self.state == AgentState.SELF_EVALUATION:
                    self.transition_to_state(AgentState.IDLE)
                    break
                
                # pursue goal
                self.adopt_active_goals()

                next_state = self.identify_next_state()

                self.transition_to_state(next_state)

            return self.analysis_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing news item: {str(e)}")
            # when automated plan rules fail, fallback to procedural state transition
            self.procedural_state_transition()
            raise

if __name__ == '__main__':
    FNA = FakeNewsAgent(OntologyService(), LLMService())
    recommendation = FNA.analyze_news_item()
    print(recommendation['final_output'])
