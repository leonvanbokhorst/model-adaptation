"""
Medical Bayesian Networks and Large Language Models: A Historical Perspective

Historical Significance:
------------------------
Bayesian networks in medicine trace back to the 1980s with systems like MYCIN and 
INTERNIST-1. These early expert systems demonstrated both the potential and limitations 
of rule-based medical reasoning:

1. Early Systems (1970s-1980s):
   - MYCIN: Used certainty factors for bacterial infections
   - INTERNIST-1: Attempted comprehensive internal medicine diagnosis
   - Key limitation: Rigid, rule-based reasoning

2. Bayesian Revolution (1990s):
   - Introduction of probabilistic graphical models
   - QMR-DT: First major Bayesian medical diagnosis system
   - Enabled handling of uncertainty and incomplete information

3. Modern Integration (2020s):
   - Combination of Bayesian networks with LLMs
   - Natural language understanding meets probabilistic reasoning
   - Explainable AI becomes crucial for medical applications

Key Innovations in This Implementation:
-------------------------------------
1. Hybrid Architecture:
   - Bayesian networks provide probabilistic reasoning
   - LLMs enable natural language understanding
   - Combines structured and unstructured data processing

2. Explainability:
   - Every decision has a traceable reasoning path
   - Natural language explanations for medical professionals
   - Audit trail for accountability

3. Medical Knowledge Integration:
   - Dynamic knowledge structure creation
   - Causal relationship extraction
   - Evidence-based reasoning paths

Technical Components:
-------------------
1. Bayesian Network:
   - Nodes: Medical conditions/symptoms
   - Edges: Causal relationships
   - CPTs: Conditional probabilities

2. LLM Integration:
   - Structure learning from text
   - Evidence extraction
   - Natural language generation

3. Logging System:
   - Diagnostic process tracking
   - Decision auditing
   - Quality control

This system represents a step toward more interpretable and reliable medical AI,
addressing key challenges in healthcare automation:
- Uncertainty handling
- Decision transparency
- Knowledge integration
- Clinical workflow integration

Usage Example:
-------------
patient_story = '''
I am experiencing severe fatigue, especially in the mornings,
along with persistent headaches and occasional dizziness.
'''

system = BayesianLLM()
system.setup_medical_network(patient_story)
diagnosis = system.generate_diagnostic_reasoning(evidence)

The system will:
1. Extract relevant medical concepts
2. Build a Bayesian network structure
3. Generate probabilistic diagnoses
4. Provide natural language explanations
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, NamedTuple
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from dataclasses import dataclass
import csv
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

MODEL_NAME = "hermes3:latest"


@dataclass
class DiagnosticReasoning:
    conclusion: str
    confidence: float
    evidence_path: List[str]
    alternative_explanations: List[Tuple[str, float]]


class BayesianLLM:
    """
    A Bayesian network-based medical diagnosis system that uses LLMs for:
    1. Network structure learning
    2. Evidence extraction
    3. Diagnostic reasoning
    4. Natural language explanations

    Key Components:
    - LLM Integration: Uses Ollama for natural language understanding
    - Bayesian Network: Captures causal relationships between medical concepts
    - Logging System: Tracks diagnostic processes for accountability
    - Explanation Generation: Provides human-readable reasoning paths
    """

    def __init__(self, model_name: str = MODEL_NAME):
        """Initialize the BayesianLLM system"""
        self.model_name = model_name
        self.llm = ChatOllama(model=model_name)
        self.nodes: Dict[str, List[str]] = {}
        self.network: Optional[BayesianNetwork] = None
        self.patient_story: str = ""
        self.log_file = Path("diagnostics/diagnostic_logs.csv")
        self._initialize_log_file()

    def _initialize_log_file(self):
        """Initialize the CSV log file with headers if it doesn't exist"""
        if not self.log_file.exists():
            logger.info(f"Creating log file: {self.log_file}")
            if not self.log_file.parent.exists():
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created parent directory: {self.log_file.parent}")
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "patient_story",
                        "extracted_evidence",
                        "primary_conclusion",
                        "confidence",
                        "evidence_path",
                        "alternative_explanations",
                        "network_structure",
                    ]
                )

    def log_diagnostic_process(
        self, evidence: Dict[str, str], diagnosis: DiagnosticReasoning
    ) -> None:
        """
        Creates an audit trail of diagnostic decisions.
        
        Purpose:
        1. Accountability: Track decision-making process
        2. Learning: Analyze patterns in successful diagnoses
        3. Quality Control: Monitor system performance
        
        Stores:
        - Timestamp: When diagnosis was made
        - Patient Story: Original description
        - Evidence: What was observed
        - Reasoning: How conclusions were reached
        - Network State: System configuration
        
        This is crucial for:
        - Medical documentation
        - System improvement
        - Potential legal requirements
        """
        try:
            # Convert network structure to string representation
            network_structure = (
                [f"{cause} → {effect}" for cause, effect in self.network.edges()]
                if self.network
                else []
            )

            # Prepare the log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "patient_story": self.patient_story.strip(),
                "extracted_evidence": json.dumps(evidence),
                "primary_conclusion": diagnosis.conclusion,
                "confidence": str(diagnosis.confidence),  # Convert float to string
                "evidence_path": json.dumps(diagnosis.evidence_path),
                "alternative_explanations": json.dumps(
                    diagnosis.alternative_explanations
                ),
                "network_structure": json.dumps(network_structure),
            }

            logger.debug(f"Preparing to log entry: {log_entry}")

            # Write to CSV
            with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timestamp",
                        "patient_story",
                        "extracted_evidence",
                        "primary_conclusion",
                        "confidence",
                        "evidence_path",
                        "alternative_explanations",
                        "network_structure",
                    ],
                )

                writer.writerow(log_entry)

            logger.info(f"Successfully logged diagnostic process to {self.log_file}")

        except Exception as e:
            logger.error(f"Failed to log diagnostic process: {e}", exc_info=True)
            raise

    def create_node(self, description: str) -> Tuple[str, List[str]]:
        """
        Converts natural language descriptions into Bayesian network nodes.
        
        The Process:
        1. Takes a medical concept description (e.g., "patient's fatigue level")
        2. Uses LLM to generate:
           - A standardized node name (snake_case)
           - 5 possible states for that node
        3. Returns structured format for network building
        
        Example:
        Input: "patient's fatigue level"
        Output: ("fatigue_level", ["none", "mild", "moderate", "severe", "extreme"])
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that creates nodes for Bayesian networks. Return only valid JSON.",
                ),
                (
                    "user",
                    """Create a node for a Bayesian network based on this description:
                "{description}"
                
                Return a JSON object with:
                1. A short snake_case name for the node
                2. A list of 5 possible states for this node
                
                Return ONLY the JSON object, no additional text or formatting:
                {{"name": "node_name", "states": ["state1", "state2", "state3", "state4", "state5"]}}""",
                ),
            ]
        )

        print(f"🔄 Creating node from description: '{description}'")
        print("📤 Sending request to LLM...")

        try:
            chain = prompt | self.llm | StrOutputParser()
            content = chain.invoke({"description": description})

            # Clean up the response
            content = content.strip()
            # Remove any markdown formatting
            if "```" in content:
                content = content.split("```")[1]
                if "json" in content.split("\n")[0]:
                    content = "\n".join(content.split("\n")[1:])
            # Remove any trailing backticks
            content = content.replace("`", "").strip()

            try:
                node_info = json.loads(content)
                if (
                    not isinstance(node_info, dict)
                    or "name" not in node_info
                    or "states" not in node_info
                ):
                    raise ValueError("Invalid JSON structure")
                return node_info["name"], node_info["states"]
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {content}")
                logger.error(f"JSON error: {e}")
                raise
            except ValueError as e:
                logger.error(f"Invalid JSON structure: {content}")
                raise

        except Exception as e:
            logger.error(f"Unexpected error in create_node: {e}")
            logger.error(f"Failed description: {description}")
            raise

    def extract_relationships(self, text: str) -> List[Tuple[str, str]]:
        """
        Identifies causal relationships between medical concepts.
        
        The Process:
        1. Analyzes patient story for cause-effect relationships
        2. Maps relationships to existing network nodes
        3. Validates relationships against known nodes
        
        Example:
        "Fatigue is causing decreased activity" ->
        [("fatigue_level", "activity_level")]
        
        This forms the structure of our Bayesian network, showing how
        different medical conditions influence each other.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that identifies causal relationships in text.",
                ),
                (
                    "user",
                    """
            From this text, identify causal relationships between concepts:
            {text}
            
            Use ONLY these exact node names in your response:
            {nodes}
            
            Return a JSON array of objects with cause and effect properties:
            [
                {{"cause": "node_name1", "effect": "node_name2"}}
            ]
            """,
                ),
            ]
        )

        try:
            chain = prompt | self.llm | StrOutputParser()
            # Pass both text and nodes as variables to the prompt
            content = chain.invoke({"text": text, "nodes": list(self.nodes.keys())})

            # Handle markdown formatting if present
            if "```" in content:
                content = content.split("```")[1].strip()
                if content.startswith("json\n"):
                    content = content[5:]

            relationships = json.loads(content)

            # Map and validate relationships
            valid_relationships = [
                (rel["cause"], rel["effect"])
                for rel in relationships
                if rel["cause"] in self.nodes and rel["effect"] in self.nodes
            ]

            logger.info(f"Extracted relationships: {valid_relationships}")
            return valid_relationships

        except Exception as e:
            logger.error(f"Failed to extract relationships: {e}")
            return []

    def build_network(self):
        """Build the Bayesian network structure"""
        print("\n🔗 Building network structure...")
        self.network = BayesianNetwork()

        # Add nodes
        for node in self.nodes:
            self.network.add_node(node)

        # Add edges from relationships using the actual patient story
        relationships = self.extract_relationships(self.patient_story)
        print(f"Found {len(relationships)} relationships")

        for cause, effect in relationships:
            self.network.add_edge(cause, effect)

    def extract_medical_concepts(self, story: str) -> List[str]:
        """Extract relevant medical concepts from patient story"""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a medical expert that identifies key medical concepts.
                         Return ONLY a JSON array of descriptions, no additional text or formatting.
                         Example: ["concept1", "concept2", "concept3"]""",
                ),
                (
                    "user",
                    """From this patient story, identify all key medical concepts that should be modeled:
                       {story}
                       
                       Return ONLY the JSON array, no explanation or additional text.""",
                ),
            ]
        )

        try:
            chain = prompt | self.llm | StrOutputParser()
            content = chain.invoke({"story": story})

            # Clean up the response
            content = content.strip()

            # Remove any markdown formatting if present
            if "```json" in content:
                content = content.split("```json")[1]
            if "```" in content:
                content = content.split("```")[0]

            # Remove any trailing or leading whitespace or special characters
            content = content.strip("`\n\r\t ")

            logger.debug(f"Cleaned medical concepts response: {content}")

            try:
                concepts = json.loads(content)
                if not isinstance(concepts, list):
                    raise ValueError("Response is not a list")

                # Ensure all elements are strings
                concepts = [str(concept) for concept in concepts]

                if not concepts:
                    logger.warning(
                        "No medical concepts extracted, using fallback concepts"
                    )
                    return ["mood state", "energy level", "fatigue symptoms"]

                logger.info(f"Extracted medical concepts: {concepts}")
                return concepts

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {content}")
                logger.error(f"JSON error: {e}")
                # Provide fallback concepts
                return ["mood state", "energy level", "fatigue symptoms"]

        except Exception as e:
            logger.error(f"Failed to extract medical concepts: {e}")
            # Provide fallback concepts
            return ["mood state", "energy level", "fatigue symptoms"]

    def extract_evidence(self, story: str) -> Dict[str, str]:
        """Extract evidence from patient story matching node states"""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a medical expert that extracts patient information.",
                ),
                (
                    "user",
                    """
            From this patient story, extract relevant states for our nodes.
            Story: {story}
            
            Available nodes and states:
            {nodes_and_states}
            
            Return a JSON object mapping node names to their states based on the story.
            Only include nodes where there is clear evidence in the story.
            """,
                ),
            ]
        )

        nodes_str = "\n".join(
            [f"{name}: {states}" for name, states in self.nodes.items()]
        )
        chain = prompt | self.llm | StrOutputParser()
        content = chain.invoke({"story": story, "nodes_and_states": nodes_str})
        return json.loads(content)

    def setup_medical_network(self, story: str):
        """Set up a medical diagnosis network from patient story"""
        self.patient_story = story
        print("\n📋 Setting up medical diagnosis network from patient story...")

        # Extract concepts from story
        concepts = self.extract_medical_concepts(story)

        # Create nodes for each concept
        print("\n🏗️ Creating nodes...")
        for i, desc in enumerate(concepts, 1):
            print(f"\nNode {i}/{len(concepts)}\n")
            name, states = self.create_node(desc)
            self.nodes[name] = states
            print(f"✅ Created node: {name} with states: {states}")

        self.build_network()

    def generate_explanation(self, evidence: Dict[str, str]) -> str:
        """Generate a natural language explanation of the network state given evidence"""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a medical expert that explains Bayesian network states.
            Explain the relationships between variables and likely outcomes based on evidence.""",
                ),
                (
                    "user",
                    """Given this Bayesian network structure and evidence, explain the likely medical implications:

            Network Nodes: {nodes}
            
            Current Evidence: {evidence}
            
            Please provide:
            1. An interpretation of the evidence
            2. Likely implications for other variables
            3. Key relationships between variables that are relevant
            
            Keep the explanation clear and medical-focused.""",
                ),
            ]
        )

        try:
            chain = prompt | self.llm | StrOutputParser()

            # Format nodes for better readability
            nodes_str = "\n".join(
                [
                    f"- {name}: {', '.join(states)}"
                    for name, states in self.nodes.items()
                ]
            )

            # Format evidence for better readability
            evidence_str = "\n".join(
                [f"- {node}: {state}" for node, state in evidence.items()]
            )

            explanation = chain.invoke({"nodes": nodes_str, "evidence": evidence_str})

            return explanation

        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return "Unable to generate explanation due to an error."

    def generate_diagnostic_reasoning(
        self, evidence: Dict[str, str]
    ) -> DiagnosticReasoning:
        """
        Produces structured diagnostic analysis using LLM reasoning.
        
        The Process:
        1. Takes observed evidence (symptoms, test results, etc.)
        2. Uses network structure to understand relationships
        3. Generates:
           - Primary diagnosis with confidence
           - Step-by-step reasoning path
           - Alternative explanations with probabilities
        
        This mimics medical differential diagnosis where doctors:
        - Consider multiple possibilities
        - Weigh evidence strength
        - Rule out alternatives systematically
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a medical expert that provides detailed diagnostic reasoning.
                         You must respond ONLY with a JSON object in this exact format:
                         {{
                             "conclusion": "Primary diagnostic conclusion",
                             "confidence": 0.XX,
                             "evidence_path": ["step1", "step2", "step3"],
                             "alternative_explanations": [["alternative1", 0.XX], ["alternative2", 0.XX]],
                         }}
                         Do not include any additional text, markdown formatting, or explanations.""",
                ),
                (
                    "user",
                    """Based on this evidence and network structure, provide diagnostic reasoning:
                       Network Structure: {network_structure}
                       Evidence: {evidence}
                       Nodes and States: {nodes_states}""",
                ),
            ]
        )

        try:
            chain = prompt | self.llm | StrOutputParser()

            network_structure = [
                f"{cause} → {effect}" for cause, effect in self.network.edges()
            ]
            nodes_states = {node: states for node, states in self.nodes.items()}

            response = chain.invoke(
                {
                    "network_structure": network_structure,
                    "evidence": evidence,
                    "nodes_states": nodes_states,
                }
            )

            # Clean up the response
            response = response.strip()

            # Remove any markdown formatting if present
            if "```json" in response:
                response = response.split("```json")[1]
            if "```" in response:
                response = response.split("```")[0]

            # Remove any trailing or leading whitespace or special characters
            response = response.strip("`\n\r\t ")

            logger.debug(f"Cleaned response: {response}")

            try:
                result = json.loads(response)

                # Validate required fields
                required_fields = {
                    "conclusion",
                    "confidence",
                    "evidence_path",
                    "alternative_explanations",
                }
                if not all(field in result for field in required_fields):
                    missing = required_fields - set(result.keys())
                    raise ValueError(f"Missing required fields: {missing}")

                # Ensure confidence is float
                result["confidence"] = float(result["confidence"])

                # Ensure alternative_explanations format is correct
                result["alternative_explanations"] = [
                    [str(alt), float(conf)]
                    for alt, conf in result["alternative_explanations"]
                ]

                return DiagnosticReasoning(**result)

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {response}")
                logger.error(f"JSON error: {e}")
                # Provide a fallback response
                return DiagnosticReasoning(
                    conclusion="Unable to generate proper diagnosis due to system error",
                    confidence=0.0,
                    evidence_path=["System encountered an error in processing"],
                    alternative_explanations=[],
                )

        except Exception as e:
            logger.error(f"Failed to generate diagnostic reasoning: {e}")
            raise

    def explain_decision_path(self, diagnosis: DiagnosticReasoning) -> str:
        """Generate a human-readable explanation of the diagnostic decision path"""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a medical expert explaining diagnostic reasoning to other medical professionals.",
                ),
                (
                    "user",
                    """Create a detailed explanation of this diagnostic reasoning:
                       Conclusion: {conclusion}
                       Confidence: {confidence}
                       Evidence Path: {evidence_path}
                       Alternatives: {alternatives}
                       
                       Format the explanation with:
                       1. Primary conclusion and confidence level
                       2. Step-by-step reasoning path
                       3. Key evidence relationships
                       4. Alternative considerations""",
                ),
            ]
        )

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "conclusion": diagnosis.conclusion,
                "confidence": diagnosis.confidence,
                "evidence_path": diagnosis.evidence_path,
                "alternatives": diagnosis.alternative_explanations,
            }
        )

    def verify_log_file(self):
        """Verify that the log file exists and contains data"""
        try:
            if not self.log_file.exists():
                logger.error("Log file does not exist!")
                return False

            with open(self.log_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                logger.info(f"Log file contains {len(rows)} entries")
                if rows:
                    logger.debug(f"Last entry: {rows[-1]}")
                return True
        except Exception as e:
            logger.error(f"Error verifying log file: {e}", exc_info=True)
            return False


def main():
    try:
        print("\n🚀 Initializing BayesianLLM system...")
        llm = BayesianLLM()

        patient_story = """
        I am Lydia and I'm not feeling well. I feel so somber and tired.
        """

        # Store patient story
        llm.patient_story = patient_story

        # Setup network
        llm.setup_medical_network(patient_story)

        # Extract evidence from story
        evidence = llm.extract_evidence(patient_story)
        logger.info(f"Extracted Evidence: {evidence}")

        # Generate detailed diagnostic reasoning
        diagnosis = llm.generate_diagnostic_reasoning(evidence)
        logger.info(f"Generated diagnosis: {diagnosis}")

        # Log the diagnostic process
        llm.log_diagnostic_process(evidence, diagnosis)

        print("\n📊 Diagnostic Analysis:")
        print(
            f"Primary Conclusion: {diagnosis.conclusion} (Confidence: {diagnosis.confidence*100:.1f}%)"
        )
        print("\nReasoning Path:")
        for step in diagnosis.evidence_path:
            print(f"- {step}")

        print("\nAlternative Explanations:")
        for alt, conf in diagnosis.alternative_explanations:
            print(f"- {alt} ({conf*100:.1f}% confidence)")

        # Generate detailed explanation
        print("\n📝 Detailed Medical Explanation:")
        explanation = llm.explain_decision_path(diagnosis)
        print(explanation)

        print(f"\n✅ Diagnostic process has been logged to: {llm.log_file}")

        # Verify the log file
        llm.verify_log_file()

    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
