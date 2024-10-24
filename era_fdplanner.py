from py2pddl import Domain, create_type
import subprocess
import os
import tempfile
from typing import List, Optional
import logging
from baseline_agent import AgentState, Goal, Plan, FakeNewsAgent 
from era_planner import ExpertReasonerAgent
from ontology_service import OntologyService
from llm_service import LLMService
from py2pddl import Domain, create_type
from py2pddl import predicate, action, goal, init

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
    

class FastDownwardPlanner():
    """Wrapper for Fast-Downward PDDL planner"""
    
    def __init__(self, fd_path: str = "fast-downward.py"):
        """
        Initialize Fast-Downward planner
        
        Args:
            fd_path: Path to fast-downward.py script
        """
        self.fd_path = fd_path
        self.logger = logging.getLogger(__name__ + '.FastDownwardPlanner')
        
    def write_pddl_files(self, problem: dict) -> tuple[str, str]:
        """
        Write PDDL domain and problem files
        
        Args:
            problem: Dictionary containing domain and problem definitions
            
        Returns:
            Tuple of (domain_file_path, problem_file_path)
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pddl', delete=False) as domain_file:
            domain_file.write(str(problem['domain']))
            domain_path = domain_file.name
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pddl', delete=False) as problem_file:
            # Write objects
            problem_file.write("(define (problem fact-check-problem)\n")
            problem_file.write(f"(:domain {problem['domain'].__class__.__name__})\n")
            
            # Write objects
            problem_file.write("(:objects\n")
            for type_name, objects in problem['objects'].items():
                objects_str = " ".join(str(obj) for obj in objects)
                problem_file.write(f"  {objects_str} - {type_name}\n")
            problem_file.write(")\n")
            
            # Write initial state
            problem_file.write("(:init\n")
            for pred in problem['init']:
                problem_file.write(f"  {str(pred)}\n")
            problem_file.write(")\n")
            
            # Write goal state
            problem_file.write("(:goal (and\n")
            for pred in problem['goal']:
                problem_file.write(f"  {str(pred)}\n")
            problem_file.write("))\n")
            
            problem_file.write(")")
            problem_path = problem_file.name
            
        return domain_path, problem_path

    def run_fast_downward(self, domain_file: str, problem_file: str) -> Optional[List[str]]:
        """
        Run Fast-Downward planner
        
        Args:
            domain_file: Path to PDDL domain file
            problem_file: Path to PDDL problem file
            
        Returns:
            List of plan steps or None if planning failed
        """
        try:
            # Run Fast-Downward with A* search and LM-Cut heuristic
            cmd = [
                self.fd_path,
                domain_file,
                problem_file,
                "--search",
                "astar(lmcut())"
            ]
            
            self.logger.info(f"Running Fast-Downward command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                self.logger.error(f"Fast-Downward failed: {result.stderr}")
                return None
                
            # Parse plan from sas_plan file
            plan_file = "sas_plan"
            if os.path.exists(plan_file):
                with open(plan_file) as f:
                    plan = []
                    for line in f:
                        if line.startswith("(") and line.endswith(")"):
                            # Extract action name and parameters
                            action = line.strip("()\n").split()[0]
                            plan.append(action)
                    return plan
            else:
                self.logger.error("No plan file generated")
                return None
                
        except Exception as e:
            self.logger.error(f"Error running Fast-Downward: {str(e)}")
            return None
            
        finally:
            # Cleanup temporary files
            for file in [domain_file, problem_file, "sas_plan"]:
                if os.path.exists(file):
                    os.remove(file)

class ExpertReasonerAgent_FDP(ExpertReasonerAgent):
    def __init__(self, ontology_service: OntologyService=None, llm_service: LLMService=None,
                 fd_path: str="fast-downward.py"):
        super().__init__(ontology_service, llm_service)
        self.planner = FastDownwardPlanner(fd_path)
        self.logger = logging.getLogger(__name__ + '.ExpertReasonerAgent')
        self.domain = FactCheckDomain()
        self.initialize_pddl_state()

    def solve_pddl_problem(self, problem: dict) -> List[str]:
        """Solve PDDL problem using Fast-Downward"""
        self.logger.info("Starting Fast-Downward planning")
        
        # Write PDDL files
        domain_file, problem_file = self.planner.write_pddl_files(problem)
        
        # Run Fast-Downward
        plan = self.planner.run_fast_downward(domain_file, problem_file)
        
        if plan:
            self.logger.info(f"Plan found with {len(plan)} steps")
            return plan
        else:
            self.logger.error("No plan found")
            return []

# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Initialize agent with path to Fast-Downward
    agent = ExpertReasonerAgent_FDP(
        OntologyService(),
        LLMService(),
        fd_path="/path/to/fast-downward.py"
    )
    
    results = agent.analyze_news_item("Does eating spicy food cause hair loss?")
    logging.info(f"Analysis results: {results}")