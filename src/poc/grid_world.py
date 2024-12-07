import os
# Set tokenizer parallelism to false to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from enum import Enum, auto
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
import subprocess

class ResourceType(Enum):
    FOOD = auto()
    WATER = auto()
    TOOL = auto()
    INFORMATION = auto()

@dataclass(frozen=True)  # Make the class immutable
class Position:
    x: int
    y: int

    def manhattan_distance(self, other: 'Position') -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        if not isinstance(other, Position):
            return NotImplemented
        return self.x == other.x and self.y == other.y

class GridWorld:
    def __init__(self, size: int = 10):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.resources: Dict[Position, ResourceType] = {}
        self.obstacles: List[Position] = []
        
    def add_resource(self, pos: Position, resource_type: ResourceType):
        if 0 <= pos.x < self.size and 0 <= pos.y < self.size:
            self.resources[pos] = resource_type
            
    def add_obstacle(self, pos: Position):
        if 0 <= pos.x < self.size and 0 <= pos.y < self.size:
            self.obstacles.append(pos)
            
    def is_valid_move(self, pos: Position) -> bool:
        return (0 <= pos.x < self.size and 
                0 <= pos.y < self.size and 
                pos not in self.obstacles)

class AgentState:
    def __init__(self):
        self.health = 100.0
        self.energy = 100.0
        self.inventory: Dict[ResourceType, int] = {rt: 0 for rt in ResourceType}
        self.stress = 0.0
        self.experience = 0.0
        
    def update_stress(self, delta: float):
        self.stress = max(0.0, min(100.0, self.stress + delta))
        
    def update_health(self, delta: float):
        self.health = max(0.0, min(100.0, self.health + delta))
        
    def update_energy(self, delta: float):
        self.energy = max(0.0, min(100.0, self.energy + delta))

class Action(Enum):
    MOVE_NORTH = auto()
    MOVE_SOUTH = auto()
    MOVE_EAST = auto()
    MOVE_WEST = auto()
    COLLECT = auto()
    REST = auto()
    SHARE = auto()

class AdaptiveAgent(nn.Module):
    def __init__(self, 
                 world_size: int,
                 embed_dim: int = 32,
                 learning_rate: float = 0.01):
        super().__init__()
        
        # Determine device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.to(self.device)
        
        # Initialize position near center with randomization
        center = world_size // 2
        rand_offset = 2  # Random offset from center
        x = center + np.random.randint(-rand_offset, rand_offset + 1)
        y = center + np.random.randint(-rand_offset, rand_offset + 1)
        
        # Ensure position is within bounds
        x = max(0, min(x, world_size - 1))
        y = max(0, min(y, world_size - 1))
        
        self.position = Position(x, y)
        self.state = AgentState()
        self.world_size = world_size
        self.embed_dim = embed_dim
        
        # Neural network for action selection
        self.policy_net = nn.Sequential(
            nn.Linear(embed_dim + 6, 64),  # State + position encoding
            nn.ReLU(),
            nn.Linear(64, len(Action)),
        ).to(self.device)
        
        # Value network for state evaluation
        self.value_net = nn.Sequential(
            nn.Linear(embed_dim + 6, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + 
            list(self.value_net.parameters()),
            lr=learning_rate
        )
        
        # Psychological parameters
        self.trauma_threshold = 50.0
        self.recovery_rate = 0.1
        self.adaptation_rate = 0.2
        
        # Experience memory
        self.experiences: List[Tuple[torch.Tensor, Action, float]] = []
        
    def get_state_encoding(self) -> torch.Tensor:
        """Encode agent's current state and position."""
        position_encoding = torch.tensor([
            self.position.x / self.world_size,
            self.position.y / self.world_size,
            self.state.health / 100.0,
            self.state.energy / 100.0,
            self.state.stress / 100.0,
            self.state.experience / 100.0
        ], device=self.device)
        
        return position_encoding
    
    def select_action(self, state_embedding: torch.Tensor) -> Action:
        """Select action based on current state using policy network."""
        # Ensure state_embedding is on the correct device and shape
        state_embedding = state_embedding.to(self.device)
        
        # Ensure we have a 2D tensor (batch_size, features)
        if state_embedding.dim() == 1:
            state_embedding = state_embedding.unsqueeze(0)
        elif state_embedding.dim() == 3:
            # If we have sequence dimension, take mean
            state_embedding = state_embedding.mean(dim=1)
        
        # Get state encoding and expand to match batch size
        state_encoding = self.get_state_encoding()
        state_encoding = state_encoding.expand(state_embedding.size(0), -1)
        
        # Combine embeddings
        combined_state = torch.cat([state_embedding, state_encoding], dim=1)
        
        # Get action probabilities
        action_logits = self.policy_net(combined_state)
        action_probs = torch.softmax(action_logits, dim=1)
        
        # Sample action
        action_idx = torch.multinomial(action_probs[0], 1).item()
        return Action(action_idx + 1)
    
    def update_from_experience(self, 
                             state: torch.Tensor, 
                             action: Action, 
                             reward: float):
        """Learn from experience using reinforcement learning."""
        # Ensure state tensor is on the correct device and shape
        state = state.to(self.device)
        
        # Ensure we have a 2D tensor (batch_size, features)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        elif state.dim() == 3:
            # If we have sequence dimension, take mean
            state = state.mean(dim=1)
        
        self.experiences.append((state, action, reward))
        
        # Update stress based on experience
        if reward < 0:
            self.state.update_stress(abs(reward) * 0.1)
        else:
            self.state.update_stress(-reward * 0.05)
            
        # Trigger trauma response if stress exceeds threshold
        if self.state.stress > self.trauma_threshold:
            self.adaptation_rate *= 0.9  # Reduced adaptation when traumatized
            
        # Update experience points
        self.state.experience += abs(reward) * 0.1
        
        # Periodic learning from experiences
        if len(self.experiences) >= 10:
            self._batch_learn()
            self.experiences.clear()
            
    def _batch_learn(self):
        """Perform batch learning from collected experiences."""
        if not self.experiences:
            return
            
        states, actions, rewards = zip(*self.experiences)
        
        # Stack states and ensure proper shape
        state_tensor = torch.cat([s for s in states])
        action_tensor = torch.tensor([a.value for a in actions], device=self.device)
        reward_tensor = torch.tensor(rewards, device=self.device)
        
        # Get state encodings for all experiences
        state_encoding = self.get_state_encoding()
        state_encodings = state_encoding.expand(len(states), -1)
        
        # Combine with state encodings
        combined_states = torch.cat([state_tensor, state_encodings], dim=1)
        
        # Compute value predictions
        state_values = self.value_net(combined_states).squeeze()
        
        # Compute advantages
        advantages = reward_tensor - state_values.detach()
        
        # Policy loss
        action_probs = self.policy_net(combined_states)
        policy_loss = -torch.mean(
            torch.log_softmax(action_probs, dim=1).gather(1, action_tensor.unsqueeze(1)) 
            * advantages.unsqueeze(1)
        )
        
        # Value loss
        value_loss = torch.mean((reward_tensor - state_values) ** 2)
        
        # Combined loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

class Simulation:
    def __init__(self, 
                 world_size: int = 10,
                 num_agents: int = 2,
                 embed_dim: int = 32):
        self.world = GridWorld(world_size)
        self.agents = [AdaptiveAgent(world_size, embed_dim) for _ in range(num_agents)]
        self.text_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.device = self.agents[0].device  # Use same device as agents
        
        # Add embedding reducer to match expected dimensions
        self.embedding_reducer = nn.Sequential(
            nn.Linear(384, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        ).to(self.device)
        
        # Initialize world with resources and obstacles
        self._setup_environment()
        
    def _setup_environment(self):
        """Initialize the world with resources and obstacles."""
        center = self.world.size // 2
        
        # Add resources in a somewhat circular pattern around center
        for _ in range(self.world.size * 2):  # More resources
            # Random angle and distance from center
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.normal(self.world.size / 4, self.world.size / 8)
            
            # Convert to grid coordinates
            x = int(center + distance * np.cos(angle))
            y = int(center + distance * np.sin(angle))
            
            # Ensure within bounds
            x = max(0, min(x, self.world.size - 1))
            y = max(0, min(y, self.world.size - 1))
            
            pos = Position(x, y)
            resource = np.random.choice(list(ResourceType))
            self.world.add_resource(pos, resource)
        
        # Add obstacles in a sparse pattern
        for _ in range(self.world.size):
            # Random position with slight bias towards edges
            if np.random.random() < 0.7:  # 70% chance for edge obstacles
                if np.random.random() < 0.5:
                    x = np.random.randint(0, self.world.size)
                    y = np.random.choice([0, 1, self.world.size-2, self.world.size-1])
                else:
                    x = np.random.choice([0, 1, self.world.size-2, self.world.size-1])
                    y = np.random.randint(0, self.world.size)
            else:  # 30% chance for random obstacles
                x = np.random.randint(0, self.world.size)
                y = np.random.randint(0, self.world.size)
            
            # Don't place obstacles too close to center
            if abs(x - center) <= 2 and abs(y - center) <= 2:
                continue
            
            pos = Position(x, y)
            self.world.add_obstacle(pos)
    
    def get_llm_advice(self, agent: AdaptiveAgent) -> str:
        """Get strategic advice from LLM based on agent's current state."""
        # Find nearby resources
        nearby_resources = [
            (pos, resource_type) 
            for pos, resource_type in self.world.resources.items()
            if pos.manhattan_distance(agent.position) <= 2
        ]
        
        # Only get LLM advice every few steps or when needed
        if agent.state.energy < 30 or agent.state.stress > 70 or len(agent.experiences) >= 8:
            prompt = f"""
            Agent Status:
            - Position: ({agent.position.x}, {agent.position.y})
            - Health: {agent.state.health:.1f}
            - Energy: {agent.state.energy:.1f}
            - Stress: {agent.state.stress:.1f}
            - Experience: {agent.state.experience:.1f}
            - Inventory: {agent.state.inventory}
            
            World State:
            - Size: {self.world.size}x{self.world.size}
            - Nearby resources: {[(pos, r.name) for pos, r in nearby_resources]}
            
            Based on this situation, what strategic actions should the agent take?
            Focus on concrete, actionable advice for resource gathering and survival.
            Keep response brief and focused on immediate actions.
            """
            
            try:
                result = subprocess.run(
                    ["ollama", "run", "llama3.2:latest"],
                    input=prompt.encode("utf-8"),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5,  # Reduced timeout
                )
                result.check_returncode()
                return result.stdout.decode("utf-8").strip()
            except Exception as e:
                print(f"LLM advice error: {e}")
                # Return simple heuristic-based advice instead
                if agent.state.energy < 30:
                    return "Find nearest resource to restore energy"
                elif agent.state.stress > 70:
                    return "Take rest action to reduce stress"
                else:
                    return "Continue exploring and gathering resources"
        else:
            # Use previous advice or simple heuristics
            if agent.state.energy < 50:
                return "Conserve energy and gather nearby resources"
            elif nearby_resources:  # Check if there are any nearby resources
                return "Collect available resources"
            else:
                return "Explore new areas"
            
    def process_text_embedding(self, text: str) -> torch.Tensor:
        """Process text to embedding with proper dimensions."""
        # Get embedding from sentence transformer
        with torch.no_grad():
            embedding = self.text_embedder.encode(
                text,
                convert_to_tensor=True,
                device=self.device
            )
            
            # Add batch dimension if needed
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
                
            # Reduce embedding dimension
            reduced = self.embedding_reducer(embedding)
            
            return reduced
    
    def step(self) -> List[float]:
        """Simulate one step of the environment."""
        rewards = []
        
        for agent in self.agents:
            # Get LLM advice
            advice = self.get_llm_advice(agent)
            
            # Process advice into embedding
            advice_embedding = self.process_text_embedding(advice)
            
            # Select and execute action
            action = agent.select_action(advice_embedding)
            reward = self._execute_action(agent, action)
            
            # Update agent state
            agent.update_from_experience(advice_embedding, action, reward)
            rewards.append(reward)
            
        return rewards
    
    def _execute_action(self, agent: AdaptiveAgent, action: Action) -> float:
        """Execute action and return reward."""
        reward = 0.0
        new_pos = Position(agent.position.x, agent.position.y)
        
        # Movement actions
        if action == Action.MOVE_NORTH:
            new_pos = Position(new_pos.x, min(new_pos.y + 1, self.world.size - 1))
        elif action == Action.MOVE_SOUTH:
            new_pos = Position(new_pos.x, max(new_pos.y - 1, 0))
        elif action == Action.MOVE_EAST:
            new_pos = Position(min(new_pos.x + 1, self.world.size - 1), new_pos.y)
        elif action == Action.MOVE_WEST:
            new_pos = Position(max(new_pos.x - 1, 0), new_pos.y)
            
        # Check if move is valid
        if self.world.is_valid_move(new_pos):
            agent.position = new_pos
            agent.state.update_energy(-1.0)  # Movement costs energy
        else:
            reward -= 5.0  # Penalty for invalid move
            
        # Resource collection
        if action == Action.COLLECT:
            if agent.position in self.world.resources:
                resource = self.world.resources[agent.position]
                agent.state.inventory[resource] += 1
                reward += 10.0
                agent.state.update_energy(-2.0)
                del self.world.resources[agent.position]
                
        # Rest action
        elif action == Action.REST:
            agent.state.update_energy(5.0)
            agent.state.update_stress(-2.0)
            reward += 2.0
            
        # Basic survival mechanics
        agent.state.update_energy(-0.5)  # Basic energy consumption
        if agent.state.energy < 20.0:
            reward -= 2.0
            
        return reward
