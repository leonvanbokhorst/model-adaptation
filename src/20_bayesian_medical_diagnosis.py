import json
import logging
from typing import Dict, List, Tuple, Optional
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "hermes3:latest"


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

        # Add edges from relationships
        medical_text = """
        In medical diagnosis, patient age and overall health status influence symptom severity.
        The presence of fever often leads to fatigue symptoms.
        Both symptoms and test results help determine the final diagnosis.
        """

        relationships = self.extract_relationships(medical_text)
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
            4. A list of likely diagnoses
            5. Severity of symptoms
            6. Suggestions for next steps (if needed)
            
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
    
    print("\n📝 Generating explanation...")
    explanation = llm.generate_explanation(evidence)
    print(f"\nExplanation:\n{explanation}")


if __name__ == "__main__":
    main()
