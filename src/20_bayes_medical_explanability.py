import json
import logging
from typing import Dict, List, Tuple, Optional, NamedTuple
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "hermes3:latest"

@dataclass
class DiagnosticReasoning:
    conclusion: str
    confidence: float
    evidence_path: List[str]
    alternative_explanations: List[Tuple[str, float]]
    supporting_literature: List[str]

class BayesianLLM:
    def __init__(self, model_name: str = MODEL_NAME):
        """Initialize the BayesianLLM system"""
        self.model_name = model_name
        self.llm = ChatOllama(model=model_name)
        self.nodes: Dict[str, List[str]] = {}
        self.network: Optional[BayesianNetwork] = None
        self.patient_story: str = ""

    def create_node(self, description: str) -> Tuple[str, List[str]]:
        """Create a node with states based on LLM description"""
        prompt = ChatPromptTemplate.from_messages([
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
        ])

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
                if not isinstance(node_info, dict) or "name" not in node_info or "states" not in node_info:
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
        """Extract causal relationships between nodes"""
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
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical expert that identifies key medical concepts."),
            ("user", """
            From this patient story, identify all key medical concepts that should be modeled:
            {story}
            
            Return a JSON array of descriptions:
            ["concept1", "concept2", "concept3", "concept4", "concept5"]
            """)
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        content = chain.invoke({"story": story})
        return json.loads(content)

    def extract_evidence(self, story: str) -> Dict[str, str]:
        """Extract evidence from patient story matching node states"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical expert that extracts patient information."),
            ("user", """
            From this patient story, extract relevant states for our nodes.
            Story: {story}
            
            Available nodes and states:
            {nodes_and_states}
            
            Return a JSON object mapping node names to their states based on the story.
            Only include nodes where there is clear evidence in the story.
            """)
        ])
        
        nodes_str = "\n".join([f"{name}: {states}" for name, states in self.nodes.items()])
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

    def generate_diagnostic_reasoning(self, evidence: Dict[str, str]) -> DiagnosticReasoning:
        """Generate detailed diagnostic reasoning with evidence paths and confidence levels"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical expert that provides detailed diagnostic reasoning.
                         Format your response as JSON with the following structure:
                         {{
                             "conclusion": "Primary diagnostic conclusion",
                             "confidence": 0.XX,
                             "evidence_path": ["step1", "step2", "step3"],
                             "alternative_explanations": [["alternative1", 0.XX], ["alternative2", 0.XX]],
                             "supporting_literature": ["reference1", "reference2"]
                         }}"""),
            ("user", """Given this Bayesian network and evidence, provide detailed diagnostic reasoning:
                       Network Structure: {network_structure}
                       Evidence: {evidence}
                       Nodes and States: {nodes_states}
                       
                       Provide step-by-step reasoning, confidence levels, and alternative explanations.""")
        ])

        try:
            chain = prompt | self.llm | StrOutputParser()
            
            # Format network structure
            network_structure = [f"{cause} → {effect}" for cause, effect in self.network.edges()]
            nodes_states = {node: states for node, states in self.nodes.items()}
            
            result = json.loads(chain.invoke({
                "network_structure": network_structure,
                "evidence": evidence,
                "nodes_states": nodes_states
            }))
            
            return DiagnosticReasoning(**result)
        except Exception as e:
            logger.error(f"Failed to generate diagnostic reasoning: {e}")
            raise

    def explain_decision_path(self, diagnosis: DiagnosticReasoning) -> str:
        """Generate a human-readable explanation of the diagnostic decision path"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical expert explaining diagnostic reasoning to other medical professionals."),
            ("user", """Create a detailed explanation of this diagnostic reasoning:
                       Conclusion: {conclusion}
                       Confidence: {confidence}
                       Evidence Path: {evidence_path}
                       Alternatives: {alternatives}
                       
                       Format the explanation with:
                       1. Primary conclusion and confidence level
                       2. Step-by-step reasoning path
                       3. Key evidence relationships
                       4. Alternative considerations
                       5. Relevant medical literature""")
        ])

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "conclusion": diagnosis.conclusion,
            "confidence": diagnosis.confidence,
            "evidence_path": diagnosis.evidence_path,
            "alternatives": diagnosis.alternative_explanations
        })


def main():
    print("\n🚀 Initializing BayesianLLM system...")
    llm = BayesianLLM()
    
    patient_story = """
    I am Lydia and I'm not feeling well. I feel so somber and tired.
    """
    
    llm.setup_medical_network(patient_story)
    
    # Extract evidence from story
    evidence = llm.extract_evidence(patient_story)
    print(f"\nExtracted Evidence: {evidence}")
    
    # Generate detailed diagnostic reasoning
    diagnosis = llm.generate_diagnostic_reasoning(evidence)
    print("\n📊 Diagnostic Analysis:")
    print(f"Primary Conclusion: {diagnosis.conclusion} (Confidence: {diagnosis.confidence*100:.1f}%)")
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


if __name__ == "__main__":
    main()
