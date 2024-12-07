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
    FOOD = {
        'max_quantity': 5,
        'regen_rate': 0.1,
        'name': 'FOOD'
    }
    WATER = {
        'max_quantity': 10,
        'regen_rate': 0.2,
        'name': 'WATER'
    }
    TOOL = {
        'max_quantity': 3,
        'regen_rate': 0.05,
        'name': 'TOOL'
    }
    INFORMATION = {
        'max_quantity': 2,
        'regen_rate': 0.0,
        'name': 'INFORMATION'
    }
    
    @property
    def max_quantity(self) -> int:
        return self.value['max_quantity']
    
    @property
    def regen_rate(self) -> float:
        return self.value['regen_rate']
    
    @property
    def name(self) -> str:
        return self.value['name']

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

class Resource:
    def __init__(self, type: ResourceType, quantity: Optional[int] = None):
        self.type = type
        self.quantity = quantity if quantity is not None else type.max_quantity
        self.max_quantity = type.max_quantity
        self.regen_rate = type.regen_rate
    
    def consume(self, amount: int = 1) -> int:
        """Consume resource and return amount actually consumed."""
        consumed = min(self.quantity, amount)
        self.quantity -= consumed
        return consumed
    
    def regenerate(self):
        """Natural regeneration of resource."""
        if self.quantity < self.max_quantity:
            self.quantity = min(self.max_quantity, 
                              self.quantity + self.regen_rate)

class GridWorld:
    def __init__(self, size: int = 10):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.resources: Dict[Position, Resource] = {}
        self.obstacles: List[Position] = []
        
    def add_resource(self, pos: Position, resource_type: ResourceType):
        """Add a new resource with quantity."""
        if 0 <= pos.x < self.size and 0 <= pos.y < self.size:
            self.resources[pos] = Resource(resource_type)
    
    def add_obstacle(self, pos: Position):
        if 0 <= pos.x < self.size and 0 <= pos.y < self.size:
            self.obstacles.append(pos)
            
    def is_valid_move(self, pos: Position) -> bool:
        return (0 <= pos.x < self.size and 
                0 <= pos.y < self.size and 
                pos not in self.obstacles)
    
    def regenerate_resources(self):
        """Regenerate all resources by their natural rate."""
        for resource in self.resources.values():
            resource.regenerate()
    
    def remove_depleted_resources(self) -> List[Position]:
        """Remove fully depleted resources and return their positions."""
        depleted = [pos for pos, res in self.resources.items() 
                   if res.quantity <= 0]
        for pos in depleted:
            del self.resources[pos]
        return depleted

class AgentState:
    def __init__(self, parent_genes=None):
        self.health = 100.0
        self.energy = 100.0
        self.inventory: Dict[ResourceType, int] = {rt: 0 for rt in ResourceType}
        self.stress = 0.0
        self.experience = 0.0
        
        # Health and energy thresholds
        self.critical_energy = 20.0
        self.critical_health = 30.0
        self.max_health = 100.0
        self.max_energy = 100.0
        self.is_alive = True
        
        # Reproduction related attributes
        self.reproduction_cooldown = 0
        self.min_reproduction_cooldown = 50  # Steps before can reproduce again
        self.reproduction_energy_threshold = 70.0  # Minimum energy needed
        self.reproduction_health_threshold = 80.0  # Minimum health needed
        self.reproduction_cost_energy = 30.0  # Energy cost to reproduce
        self.reproduction_cost_health = 20.0  # Health cost to reproduce
        
        # Genetic traits (inherited or randomized)
        self.genes = parent_genes if parent_genes else {
            'energy_efficiency': np.random.normal(1.0, 0.1),  # How efficiently energy is used
            'health_recovery': np.random.normal(1.0, 0.1),    # Rate of health recovery
            'stress_resistance': np.random.normal(1.0, 0.1),  # Resistance to stress
            'learning_rate': np.random.normal(1.0, 0.1),      # How quickly experience is gained
        }
    
    def can_reproduce(self) -> bool:
        """Check if agent is capable of reproduction."""
        return (self.is_alive and
                self.reproduction_cooldown <= 0 and
                self.energy >= self.reproduction_energy_threshold and
                self.health >= self.reproduction_health_threshold)
    
    def reproduce_with(self, partner: 'AgentState') -> Optional[Dict]:
        """Attempt reproduction with another agent. Returns child genes if successful."""
        if not (self.can_reproduce() and partner.can_reproduce()):
            return None
        
        # Apply reproduction costs
        self.update_energy(-self.reproduction_cost_energy)
        self.update_health(-self.reproduction_cost_health)
        partner.update_energy(-partner.reproduction_cost_energy)
        partner.update_health(-partner.reproduction_cost_health)
        
        # Reset reproduction cooldowns
        self.reproduction_cooldown = self.min_reproduction_cooldown
        partner.reproduction_cooldown = partner.min_reproduction_cooldown
        
        # Create child genes through genetic crossover and mutation
        child_genes = {}
        for trait in self.genes:
            # Crossover: randomly select gene from either parent
            parent_gene = self.genes[trait] if np.random.random() < 0.5 else partner.genes[trait]
            # Mutation: small random adjustment
            mutation = np.random.normal(0, 0.05)  # 5% mutation rate
            child_genes[trait] = max(0.5, min(1.5, parent_gene + mutation))  # Clamp to reasonable range
        
        return child_genes
    
    def update_reproduction_state(self):
        """Update reproduction-related state variables."""
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1
    
    def update_stress(self, delta: float):
        """Update stress level with bounds."""
        self.stress = max(0.0, min(100.0, self.stress + delta))
        
    def update_health(self, delta: float):
        """Update health with bounds and check death condition."""
        old_health = self.health
        self.health = max(0.0, min(self.max_health, self.health + delta))
        
        # Check death condition
        if self.health <= 0:
            self.is_alive = False
            
        return self.health - old_health
        
    def update_energy(self, delta: float):
        """Update energy with bounds and health impact."""
        old_energy = self.energy
        self.energy = max(0.0, min(self.max_energy, self.energy + delta))
        
        # Very low energy damages health
        if self.energy <= self.critical_energy:
            health_damage = -0.5 * (1.0 - self.energy / self.critical_energy)
            self.update_health(health_damage)
            
        return self.energy - old_energy
    
    def natural_decay(self):
        """Apply natural state decay per timestep."""
        decay_effects = {
            'health': 0.0,
            'energy': -0.2  # Base energy decay per step
        }
        
        # Energy affects health
        if self.energy < self.critical_energy:
            decay_effects['health'] -= 0.5 * (1.0 - self.energy / self.critical_energy)
        
        # Stress affects both health and energy
        if self.stress > 70:
            stress_factor = (self.stress - 70) / 30  # 0 to 1 scale for stress over 70
            decay_effects['health'] -= 0.2 * stress_factor
            decay_effects['energy'] -= 0.2 * stress_factor
        
        # Apply decay effects
        health_change = self.update_health(decay_effects['health'])
        energy_change = self.update_energy(decay_effects['energy'])
        
        # Natural stress recovery
        if self.stress > 0:
            self.update_stress(-0.1)
        
        return health_change, energy_change

class Action(Enum):
    MOVE_NORTH = 0
    MOVE_SOUTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3
    COLLECT = 4
    REST = 5
    SHARE = 6

class AdaptiveAgent(nn.Module):
    def __init__(self, 
                 world_size: int,
                 embed_dim: int = 32,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,  # Discount factor
                 epsilon: float = 0.2,  # PPO clipping parameter
                 max_episode_steps: int = 100):  # Maximum steps per episode
        super().__init__()
        
        # Determine device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.to(self.device)
        
        # Initialize position near center with randomization
        center = world_size // 2
        rand_offset = 2
        x = center + np.random.randint(-rand_offset, rand_offset + 1)
        y = center + np.random.randint(-rand_offset, rand_offset + 1)
        x = max(0, min(x, world_size - 1))
        y = max(0, min(y, world_size - 1))
        
        self.position = Position(x, y)
        self.state = AgentState()
        self.world_size = world_size
        self.embed_dim = embed_dim
        
        # PPO and episode parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_episode_steps = max_episode_steps
        self.current_episode_step = 0
        self.episode_rewards = []
        self.episode_states = []
        self.episode_actions = []
        self.episode_values = []
        self.episode_log_probs = []
        
        # Neural networks for PPO
        self.policy_net = nn.Sequential(
            nn.Linear(embed_dim + 6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, len(Action))
        ).to(self.device)
        
        self.value_net = nn.Sequential(
            nn.Linear(embed_dim + 6, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + 
            list(self.value_net.parameters()),
            lr=learning_rate
        )
        
        # Psychological parameters with clearer mechanics
        self.trauma_threshold = 50.0
        self.recovery_rate = 0.1
        self.adaptation_rate = 0.2
        self.stress_decay = 0.95  # Stress decay per step
        
        # Episode goal tracking
        self.current_goal = None
        self.goal_achieved = False
        
    def reset_episode(self):
        """Reset episode-specific variables."""
        self.current_episode_step = 0
        self.episode_rewards.clear()
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_values.clear()
        self.episode_log_probs.clear()
        self.goal_achieved = False
        
        # Set new episode goal
        self._set_new_goal()
        
    def _set_new_goal(self):
        """Set a new goal for the episode based on agent's state."""
        if self.state.energy < 50:
            self.current_goal = ("energy", "Find and collect WATER to restore energy")
        elif self.state.health < 50:
            self.current_goal = ("health", "Find and collect FOOD to restore health")
        else:
            # Exploration and resource gathering goal
            self.current_goal = ("explore", "Explore and gather resources while maintaining health and energy")
    
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
    
    def select_action(self, state_embedding: torch.Tensor) -> Tuple[Action, torch.Tensor, torch.Tensor]:
        """Select action using current policy and return action, log prob, and value."""
        state_embedding = state_embedding.to(self.device)
        
        if state_embedding.dim() == 1:
            state_embedding = state_embedding.unsqueeze(0)
        elif state_embedding.dim() == 3:
            state_embedding = state_embedding.mean(dim=1)
        
        state_encoding = self.get_state_encoding()
        state_encoding = state_encoding.expand(state_embedding.size(0), -1)
        combined_state = torch.cat([state_embedding, state_encoding], dim=1)
        
        # Get action distribution
        action_logits = self.policy_net(combined_state)
        action_probs = torch.softmax(action_logits, dim=1)
        dist = torch.distributions.Categorical(action_probs)
        
        # Sample action and get log probability
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        
        # Get state value
        value = self.value_net(combined_state)
        
        return Action(action_idx.item()), log_prob, value
    
    def update_from_experience(self, state: torch.Tensor, action: Action, reward: float):
        """Store experience in episode buffer and update psychological state."""
        self.current_episode_step += 1
        
        # Store experience
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        
        # Update psychological state
        if reward < 0:
            self.state.update_stress(abs(reward) * 0.1)
        else:
            self.state.update_stress(-reward * 0.05)
        
        # Apply stress decay
        self.state.stress *= self.stress_decay
        
        # Update trauma response
        if self.state.stress > self.trauma_threshold:
            self.adaptation_rate *= 0.9
        
        # Update experience points
        self.state.experience += abs(reward) * 0.1
        
        # Check if episode should end
        if (self.current_episode_step >= self.max_episode_steps or
            self.goal_achieved or
            self.state.health <= 0 or
            self.state.energy <= 0):
            self._end_episode()
    
    def _end_episode(self):
        """Process episode data and perform PPO update."""
        if len(self.episode_rewards) < 2:
            return
        
        # Calculate returns and advantages
        returns = []
        advantages = []
        next_value = 0
        next_advantage = 0
        
        for reward in reversed(self.episode_rewards):
            next_value = reward + self.gamma * next_value
            returns.insert(0, next_value)
            
            # GAE calculation
            delta = reward + self.gamma * next_value - self.episode_values[-1].item()
            next_advantage = delta + self.gamma * 0.95 * next_advantage
            advantages.insert(0, next_advantage)
        
        # Convert to tensors and ensure proper dimensions
        states = torch.stack(self.episode_states)  # Shape: [T, B, E]
        if states.dim() == 3:
            states = states.squeeze(1)  # Remove batch dimension if present
        
        # Detach states to avoid backprop through previous updates
        states = states.detach()
        
        # Convert actions to tensor using their values (now 0-based)
        actions = torch.tensor([a.value for a in self.episode_actions], device=self.device)
        returns = torch.tensor(returns, device=self.device)
        advantages = torch.tensor(advantages, device=self.device)
        old_log_probs = torch.stack(self.episode_log_probs).detach()  # Detach old log probs
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update (multiple epochs)
        for update_epoch in range(4):
            # Get current policy distribution
            state_encoding = self.get_state_encoding()  # Shape: [F]
            state_encoding = state_encoding.unsqueeze(0).expand(states.size(0), -1)  # Shape: [T, F]
            
            # Ensure states and state_encoding have compatible dimensions for concatenation
            if states.dim() != state_encoding.dim():
                if states.dim() == 3:
                    states = states.squeeze(1)  # Remove batch dimension if present
                elif state_encoding.dim() == 3:
                    state_encoding = state_encoding.squeeze(1)
            
            combined_states = torch.cat([states, state_encoding], dim=1)
            
            # Get current policy distribution and values
            action_logits = self.policy_net(combined_states)
            action_probs = torch.softmax(action_logits, dim=1)
            dist = torch.distributions.Categorical(action_probs)
            
            # Get current log probs and values
            new_log_probs = dist.log_prob(actions)
            values = self.value_net(combined_states).squeeze()
            
            # Calculate ratios and surrogate objectives
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            # Calculate losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (returns - values).pow(2).mean()
            entropy_loss = -0.01 * dist.entropy().mean()
            
            # Combined loss
            total_loss = policy_loss + value_loss + entropy_loss
            
            # Optimization step
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()
            
            # Clear computation graph after each epoch
            del action_logits, action_probs, dist, new_log_probs, values
            del ratios, surr1, surr2, policy_loss, value_loss, entropy_loss, total_loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Reset episode data
        self.reset_episode()

class Simulation:
    def __init__(self, 
                 world_size: int = 10,
                 num_agents: int = 2,
                 embed_dim: int = 32,
                 resource_respawn_rate: float = 0.05,
                 max_agents: int = 10):  # Maximum number of agents allowed
        self.world = GridWorld(size=world_size)
        self.agents = [
            AdaptiveAgent(world_size=world_size, embed_dim=embed_dim)
            for _ in range(num_agents)
        ]
        self.embed_dim = embed_dim
        self.resource_respawn_rate = resource_respawn_rate
        self.llm_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.max_agents = max_agents
        
        # Initialize environment
        self._setup_environment()
        
        # Episode tracking
        self.current_step = 0
        self.episode_count = 0
        self.total_rewards = []
        
    def _setup_environment(self):
        """Initialize the grid world with resources and obstacles."""
        world_size = self.world.size
        num_resources = world_size * 2  # Scale resources with world size
        
        # Resource distribution parameters
        resource_weights = {
            ResourceType.FOOD: 0.3,
            ResourceType.WATER: 0.3,
            ResourceType.TOOL: 0.2,
            ResourceType.INFORMATION: 0.2
        }
        
        # Place resources in clusters
        for resource_type, weight in resource_weights.items():
            num_clusters = int(num_resources * weight) // 3
            resources_per_cluster = 3
            
            for _ in range(num_clusters):
                # Choose cluster center
                center_x = np.random.randint(0, world_size)
                center_y = np.random.randint(0, world_size)
                
                # Place resources around center
                for _ in range(resources_per_cluster):
                    offset_x = np.random.randint(-2, 3)
                    offset_y = np.random.randint(-2, 3)
                    x = max(0, min(world_size - 1, center_x + offset_x))
                    y = max(0, min(world_size - 1, center_y + offset_y))
                    pos = Position(x, y)
                    
                    if pos not in self.world.resources:
                        self.world.add_resource(pos, resource_type)
        
        # Add obstacles as barriers between resource clusters
        num_obstacles = world_size // 2
        for _ in range(num_obstacles):
            x = np.random.randint(0, world_size)
            y = np.random.randint(0, world_size)
            pos = Position(x, y)
            if pos not in self.world.resources:
                self.world.add_obstacle(pos)
    
    def get_llm_advice(self, agent: AdaptiveAgent) -> str:
        """Get context-aware advice from LLM based on agent's state and surroundings."""
        # Get nearby resources within vision range
        vision_range = 3
        nearby_resources = []
        for pos, resource in self.world.resources.items():
            if agent.position.manhattan_distance(pos) <= vision_range:
                nearby_resources.append((pos, resource))
        
        # Construct situation description
        situation = f"""
        Agent Status:
        - Health: {agent.state.health:.1f}
        - Energy: {agent.state.energy:.1f}
        - Stress: {agent.state.stress:.1f}
        - Current Goal: {agent.current_goal[1] if agent.current_goal else 'None'}
        
        Environment:
        - Position: ({agent.position.x}, {agent.position.y})
        - Nearby Resources: {len(nearby_resources)}
        """
        
        if nearby_resources:
            situation += "\nResources in view:\n"
            for pos, resource in nearby_resources:
                situation += f"- {resource.type.name} (Quantity: {resource.quantity}) at ({pos.x}, {pos.y})\n"
        
        # Generate focused advice based on agent's goal and state
        if agent.current_goal:
            goal_type, goal_desc = agent.current_goal
            if goal_type == "energy" and any(r[1].type == ResourceType.WATER for r in nearby_resources):
                return "Water source detected nearby. Move towards it to restore energy."
            elif goal_type == "health" and any(r[1].type == ResourceType.FOOD for r in nearby_resources):
                return "Food source detected nearby. Collect it to restore health."
            elif goal_type == "explore":
                if not nearby_resources:
                    return "No resources in view. Explore new areas while maintaining safe energy levels."
                else:
                    return "Multiple resources detected. Prioritize gathering based on current needs."
        
        # Default advice based on state
        if agent.state.energy < 30:
            return "Critical energy levels. Find water immediately."
        elif agent.state.health < 30:
            return "Critical health. Seek food resources."
        elif agent.state.stress > 70:
            return "High stress levels. Consider resting or finding information resources."
        
        return "Continue exploring and gathering resources efficiently."
    
    def process_text_embedding(self, text: str) -> torch.Tensor:
        """Convert LLM advice to embedding for policy network."""
        # Get text embedding
        with torch.no_grad():
            embedding = self.llm_model.encode(text, convert_to_tensor=True)
            
        # Ensure correct device and shape
        embedding = embedding.to(self.agents[0].device)
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        
        # Project to required embedding dimension if necessary
        if embedding.size(-1) != self.embed_dim:
            projection = nn.Linear(embedding.size(-1), self.embed_dim).to(self.agents[0].device)
            embedding = projection(embedding)
        
        return embedding
    
    def _handle_reproduction(self, agent: AdaptiveAgent, reward: float) -> float:
        """Handle reproduction between agents and return additional reward."""
        if not agent.state.can_reproduce() or len(self.agents) >= self.max_agents:
            return 0.0
        
        # Look for potential mates nearby
        nearby_agents = [
            other for other in self.agents
            if (other != agent and 
                other.state.can_reproduce() and
                other.position.manhattan_distance(agent.position) <= 1)
        ]
        
        if not nearby_agents:
            return 0.0
            
        # Choose the healthiest mate
        mate = max(nearby_agents, key=lambda a: a.state.health + a.state.energy)
        
        # Attempt reproduction
        child_genes = agent.state.reproduce_with(mate.state)
        if child_genes:
            # Create new agent with inherited traits
            child_agent = AdaptiveAgent(
                world_size=self.world.size,
                embed_dim=self.embed_dim
            )
            child_agent.state = AgentState(parent_genes=child_genes)
            
            # Position child near parents
            parent_pos = agent.position
            positions = [
                Position(parent_pos.x + dx, parent_pos.y + dy)
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                if self.world.is_valid_move(Position(parent_pos.x + dx, parent_pos.y + dy))
            ]
            
            if positions:
                child_agent.position = np.random.choice(positions)
                self.agents.append(child_agent)
                return 10.0  # Reward for successful reproduction
            
        return 0.0
    
    def step(self) -> List[float]:
        """Execute one step of the simulation for all agents."""
        self.current_step += 1
        rewards = []
        
        # Process each agent's turn
        for agent in self.agents:
            # Update reproduction cooldown
            agent.state.update_reproduction_state()
            
            # Get LLM advice and convert to embedding
            advice = self.get_llm_advice(agent)
            state_embedding = self.process_text_embedding(advice)
            
            # Select action using PPO
            action, log_prob, value = agent.select_action(state_embedding)
            
            # Execute action and get reward
            reward = self._execute_action(agent, action)
            
            # Handle reproduction and add reproduction reward
            reproduction_reward = self._handle_reproduction(agent, reward)
            reward += reproduction_reward
            
            # Store experience
            agent.episode_log_probs.append(log_prob)
            agent.episode_values.append(value)
            agent.update_from_experience(state_embedding, action, reward)
            
            rewards.append(reward)
        
        # Remove dead agents
        self.agents = [agent for agent in self.agents if agent.state.is_alive]
        
        # Resource management
        self.world.regenerate_resources()  # Natural regeneration
        depleted_positions = self.world.remove_depleted_resources()  # Remove depleted resources
        
        # Respawn new resources
        if np.random.random() < self.resource_respawn_rate:
            self._respawn_resources()
        
        # Add resource sharing mechanism
        def share_resources(agent1, agent2):
            if (agent1.state.health > 80 and 
                agent2.state.health < 50 and 
                agent1.state.inventory['FOOD'] > 10):
                transfer_amount = min(5, agent1.state.inventory['FOOD'])
                agent1.state.inventory['FOOD'] -= transfer_amount
                agent2.state.inventory['FOOD'] += transfer_amount
                return True
            return False
        
        # Add stress management
        def manage_stress(agent):
            if agent.state.stress > 20:
                # Implement temporary goal switch to stress reduction
                agent.current_goal = ('REST', 'Take a break to reduce stress')
                return True
            return False
        
        return rewards
    
    def _execute_action(self, agent: AdaptiveAgent, action: Action) -> float:
        """Execute action and return reward."""
        if not agent.state.is_alive:
            return -50.0  # Large penalty for actions after death
        
        reward = 0
        old_pos = agent.position
        new_pos = None
        
        # Apply natural decay
        health_change, energy_change = agent.state.natural_decay()
        reward += health_change + energy_change
        
        if not agent.state.is_alive:
            return -50.0  # Agent died from natural causes
        
        # Movement actions
        if action == Action.MOVE_NORTH:
            new_pos = Position(old_pos.x, old_pos.y - 1)
        elif action == Action.MOVE_SOUTH:
            new_pos = Position(old_pos.x, old_pos.y + 1)
        elif action == Action.MOVE_EAST:
            new_pos = Position(old_pos.x + 1, old_pos.y)
        elif action == Action.MOVE_WEST:
            new_pos = Position(old_pos.x - 1, old_pos.y)
        
        # Handle movement
        if new_pos:
            if self.world.is_valid_move(new_pos):
                agent.position = new_pos
                energy_change = agent.state.update_energy(-2)  # Movement costs more energy
                reward += energy_change
                
                # Reward for moving towards goal
                if agent.current_goal:
                    goal_type = agent.current_goal[0]
                    if goal_type in ["energy", "health"]:
                        for pos, resource in self.world.resources.items():
                            if ((goal_type == "energy" and resource.type == ResourceType.WATER) or
                                (goal_type == "health" and resource.type == ResourceType.FOOD)):
                                old_dist = old_pos.manhattan_distance(pos)
                                new_dist = new_pos.manhattan_distance(pos)
                                if new_dist < old_dist:
                                    reward += 2  # Increased reward for moving towards needed resource
            else:
                reward -= 5  # Penalty for invalid move
                agent.state.update_stress(5)  # More stress for hitting obstacles
        
        # Collection action
        elif action == Action.COLLECT:
            if agent.position in self.world.resources:
                resource = self.world.resources[agent.position]
                consumed = resource.consume()  # Try to consume 1 unit
                
                if consumed > 0:
                    agent.state.inventory[resource.type] += consumed
                    
                    # Resource-specific effects
                    if resource.type == ResourceType.FOOD:
                        health_change = agent.state.update_health(20 * consumed)  # Health gain scales with consumption
                        reward += health_change * 2
                        if agent.current_goal and agent.current_goal[0] == "health":
                            reward += 20
                            agent.goal_achieved = True
                    elif resource.type == ResourceType.WATER:
                        energy_change = agent.state.update_energy(25 * consumed)  # Energy gain scales with consumption
                        reward += energy_change * 2
                        if agent.current_goal and agent.current_goal[0] == "energy":
                            reward += 20
                            agent.goal_achieved = True
                    elif resource.type == ResourceType.TOOL:
                        reward += 5 * consumed
                        agent.state.update_stress(-10 * consumed)
                    elif resource.type == ResourceType.INFORMATION:
                        agent.state.update_stress(-15 * consumed)
                        agent.state.experience += 5 * consumed
                        reward += 5 * consumed
                    
                    # Energy cost for collecting
                    energy_change = agent.state.update_energy(-1)
                    reward += energy_change
                else:
                    reward -= 1  # Small penalty for trying to collect depleted resource
        
        # Rest action
        elif action == Action.REST:
            energy_change = agent.state.update_energy(10)
            stress_reduction = min(10, agent.state.stress)
            agent.state.update_stress(-stress_reduction)
            
            if agent.state.health < agent.state.critical_health:
                health_change = agent.state.update_health(5)
                reward += health_change * 2
            
            reward += energy_change
        
        # Share action (placeholder for multi-agent cooperation)
        elif action == Action.SHARE:
            nearby_agents = [
                other for other in self.agents
                if other != agent and other.position.manhattan_distance(agent.position) <= 1
            ]
            if nearby_agents:
                # Share resources with nearby agents
                for resource_type in ResourceType:
                    if agent.state.inventory[resource_type] > 0:
                        shared_amount = agent.state.inventory[resource_type] // 2
                        if shared_amount > 0:
                            agent.state.inventory[resource_type] -= shared_amount
                            for other in nearby_agents:
                                other.state.inventory[resource_type] += shared_amount // len(nearby_agents)
                            reward += 2 * shared_amount  # Reward for sharing
                
                agent.state.update_stress(-5)  # Stress reduction from social interaction
        
        # Check death condition
        if not agent.state.is_alive:
            reward -= 50
            agent.goal_achieved = False
        elif agent.state.health <= agent.state.critical_health:
            reward -= (agent.state.critical_health - agent.state.health) * 0.2
        
        # Update experience based on survival and rewards
        if reward > 0:
            agent.state.experience += reward * 0.1
        
        return reward
    
    def _respawn_resources(self):
        """Respawn resources with probability based on resource_respawn_rate."""
        world_size = self.world.size
        if np.random.random() < self.resource_respawn_rate:
            # Choose random position
            x = np.random.randint(0, world_size)
            y = np.random.randint(0, world_size)
            pos = Position(x, y)
            
            # Only spawn if position is empty
            if pos not in self.world.resources and pos not in self.world.obstacles:
                # Count current resources by type
                resource_counts = {rt: 0 for rt in ResourceType}
                for resource in self.world.resources.values():
                    resource_counts[resource.type] += 1
                
                # Weight towards scarcer resources
                total_resources = sum(resource_counts.values()) + 1  # Add 1 to avoid division by zero
                weights = [1 - (count / total_resources) for count in resource_counts.values()]
                weights_sum = sum(weights)
                if weights_sum > 0:  # Normalize weights
                    weights = [w / weights_sum for w in weights]
                    resource_type = np.random.choice(list(ResourceType), p=weights)
                else:
                    resource_type = np.random.choice(list(ResourceType))  # Equal probability if all weights are 0
                
                self.world.add_resource(pos, resource_type)
