import argparse
from grid_world import Simulation, AdaptiveAgent
from visualizer import SimulationVisualizer
import time
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import json
from datetime import datetime
import traceback

def normalize_metrics(values: List[float]) -> List[float]:
    """Normalize values using softmax to get relative importance."""
    values = np.array(values)
    exp_values = np.exp(values - np.max(values))  # Subtract max for numerical stability
    return exp_values / exp_values.sum()

class SimulationMetrics:
    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.goal_completion_rates: List[float] = []
        self.resource_collection_stats: Dict[str, List[int]] = {}
        self.agent_stats: List[Dict] = []
    
    def add_episode_metrics(self, 
                          rewards: List[float],
                          episode_length: int,
                          goal_completion: float,
                          resource_stats: Dict[str, int],
                          agent_stats: List[Dict]):
        """Add metrics for one episode."""
        # Calculate mean reward across all agents
        mean_reward = sum(rewards) / len(rewards) if rewards else 0
        self.episode_rewards.append(mean_reward)
        self.episode_lengths.append(episode_length)
        self.goal_completion_rates.append(goal_completion)
        
        # Update resource collection stats
        for resource_type, count in resource_stats.items():
            if resource_type not in self.resource_collection_stats:
                self.resource_collection_stats[resource_type] = []
            self.resource_collection_stats[resource_type].append(count)
        
        self.agent_stats.append(agent_stats)
    
    def save_metrics(self, output_dir: str):
        """Save metrics to JSON file."""
        metrics_data = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "goal_completion_rates": self.goal_completion_rates,
            "resource_collection_stats": self.resource_collection_stats,
            "agent_stats": self.agent_stats
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(output_dir, f"metrics_{timestamp}.json")
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        return metrics_file

class AgentAnalytics:
    """Provides comparative analysis of agent statistics."""
    
    @staticmethod
    def analyze_agents(agents: List[AdaptiveAgent]) -> Dict:
        """Analyze and compare agent statistics."""
        # Extract core metrics
        healths = [agent.state.health for agent in agents]
        energies = [agent.state.energy for agent in agents]
        stresses = [agent.state.stress for agent in agents]
        experiences = [agent.state.experience for agent in agents]
        
        # Calculate relative importance using softmax
        health_importance = normalize_metrics(healths)
        energy_importance = normalize_metrics(energies)
        stress_importance = normalize_metrics(stresses)
        exp_importance = normalize_metrics(experiences)
        
        # Calculate group statistics
        stats = {
            "group_health": {
                "mean": np.mean(healths),
                "std": np.std(healths),
                "critical_count": sum(h < 50 for h in healths)
            },
            "group_energy": {
                "mean": np.mean(energies),
                "std": np.std(energies),
                "critical_count": sum(e < 50 for e in energies)
            },
            "group_stress": {
                "mean": np.mean(stresses),
                "std": np.std(stresses),
                "high_stress_count": sum(s > 70 for s in stresses)
            },
            "resource_distribution": {},
            "goal_distribution": {}
        }
        
        # Analyze resource distribution
        all_resources = set()
        for agent in agents:
            all_resources.update(agent.state.inventory.keys())
        
        for resource in all_resources:
            resource_counts = [agent.state.inventory.get(resource, 0) for agent in agents]
            stats["resource_distribution"][resource.name] = {
                "total": sum(resource_counts),
                "mean_per_agent": np.mean(resource_counts),
                "std": np.std(resource_counts),
                "max_holder": np.argmax(resource_counts)
            }
        
        # Analyze goal distribution
        goals = [agent.current_goal[0] if agent.current_goal else "none" for agent in agents]
        goal_counts = {}
        for goal in goals:
            goal_counts[goal] = goals.count(goal)
        stats["goal_distribution"] = goal_counts
        
        return stats

def smooth_rewards(rewards: List[float], window_size: int = 5) -> List[float]:
    """Smooth rewards using moving average and clamp to reasonable range."""
    if not rewards:
        return rewards
    
    # Convert to numpy array for easier manipulation
    rewards_array = np.array(rewards)
    
    # Create moving average
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(rewards_array, kernel, mode='same')
    
    # Clamp values to prevent extreme bouncing
    MIN_REWARD = -10.0
    MAX_REWARD = 10.0
    smoothed = np.clip(smoothed, MIN_REWARD, MAX_REWARD)
    
    return smoothed.tolist()

class PopulationStats:
    def __init__(self):
        self.current_agents = 0
        self.total_born = 0
        self.total_deaths = 0
        self.previous_agents = set()  # Track agent IDs
        self.next_agent_id = 0  # Counter for generating unique agent IDs
    
    def update(self, agents: List[AdaptiveAgent]):
        """Update population statistics based on current agents."""
        # Get current set of agent IDs
        current_agent_ids = {id(agent) for agent in agents}
        
        # Check for deaths (agents that were in previous but not in current)
        deaths = len(self.previous_agents - current_agent_ids)
        self.total_deaths += deaths
        
        # Check for births (agents that are in current but not in previous)
        births = len(current_agent_ids - self.previous_agents)
        self.total_born += births
        
        # Update current count and previous set
        self.current_agents = len(agents)
        self.previous_agents = current_agent_ids
    
    def get_next_id(self) -> int:
        """Get next available agent ID."""
        id = self.next_agent_id
        self.next_agent_id += 1
        return id

def run_simulation(
    world_size: int = 6,
    num_agents: int = 5,
    num_episodes: int = 100,
    max_steps_per_episode: int = 100,
    save_interval: int = 10,
    vis_update_interval: int = 1,
    output_dir: str = "simulation_output"
):
    """Run the grid world simulation with visualization and metrics tracking."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize simulation, metrics, and population stats
    sim = Simulation(world_size=world_size, num_agents=num_agents)
    metrics = SimulationMetrics()
    pop_stats = PopulationStats()
    pop_stats.current_agents = num_agents  # Initialize with starting agents
    
    # Initialize agents with unique IDs
    agent_ids = {}  # Map agent instance to unique ID
    for agent in sim.agents:
        agent_ids[id(agent)] = pop_stats.get_next_id()
    
    try:
        # Initialize simple visualization
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.show()
        
        # Run episodes
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            episode_rewards = []
            episode_steps = 0
            goals_achieved = 0
            
            # Reset all agents for new episode
            for agent in sim.agents:
                agent.reset_episode()
            
            # Run episode steps
            for step in range(max_steps_per_episode):
                episode_steps += 1
                
                # Execute simulation step
                step_rewards = sim.step()
                episode_rewards.append(step_rewards)
                
                # Update population stats with current agents
                pop_stats.update(sim.agents)
                
                # Update agent IDs for any new agents
                for agent in sim.agents:
                    if id(agent) not in agent_ids:
                        agent_ids[id(agent)] = pop_stats.get_next_id()
                
                # Update visualization
                if step % vis_update_interval == 0:
                    try:
                        # Clear previous plot
                        ax.clear()
                        
                        # Plot grid world
                        ax.set_xlim(-1, world_size)
                        ax.set_ylim(-1, world_size)
                        ax.grid(True)
                        
                        # Add population stats to plot
                        stats_text = (
                            f"Population Stats:\n"
                            f"Current: {pop_stats.current_agents}\n"
                            f"Born: {pop_stats.total_born}\n"
                            f"Deaths: {pop_stats.total_deaths}"
                        )
                        ax.text(
                            -0.5, world_size + 0.5, stats_text,
                            fontsize=8, va='top', ha='left',
                            bbox=dict(facecolor='white', alpha=0.7)
                        )
                        
                        # Plot resources
                        for pos, resource in sim.world.resources.items():
                            color = {
                                'FOOD': 'g',
                                'WATER': 'b',
                                'TOOL': 'y',
                                'INFORMATION': 'r'
                            }.get(resource.type.name, 'k')
                            ax.plot(pos.x, pos.y, f'{color}s', markersize=8)
                        
                        # Plot agents with status indicators
                        for agent in sim.agents:
                            display_info = get_agent_display_info(agent)
                            agent_id = agent_ids[id(agent)]
                            
                            # Plot base silver circle for agent
                            ax.plot(agent.position.x, agent.position.y, 'o', 
                                   color='silver', markersize=15)
                            
                            # Add agent number (using unique ID)
                            ax.text(agent.position.x, agent.position.y, str(agent_id),
                                   color='black', ha='center', va='center',
                                   fontweight='bold', fontsize=8)
                            
                            # Add green halo for new agents
                            if display_info['is_new']:
                                ax.plot(agent.position.x, agent.position.y, 'o', 
                                       color='lime', markersize=20, alpha=0.3)
                                ax.text(agent.position.x, agent.position.y + 0.3, 'NEW!',
                                       color='green', ha='center', va='bottom',
                                       fontsize=8)
                            
                            # Add red X for dying agents
                            if display_info['is_dying']:
                                ax.plot(agent.position.x, agent.position.y, 'rx', 
                                       markersize=20, markeredgewidth=2)
                                ax.text(agent.position.x, agent.position.y - 0.3, 'DYING',
                                       color='red', ha='center', va='top',
                                       fontsize=8)
                        
                        # Update title with episode and step info
                        ax.set_title(f'Episode {episode + 1}, Step {step + 1}')
                        
                        # Refresh display
                        plt.draw()
                        plt.pause(0.01)
                        
                    except Exception as e:
                        print(f"Visualization error: {e}")
                        traceback.print_exc()
                
                # Print agent stats
                if step % vis_update_interval == 0:
                    print_agent_stats(sim.agents, step, max_steps_per_episode)
                
                # Check if all agents have completed their goals or are inactive
                all_done = all(agent.goal_achieved or 
                             agent.state.health <= 0 or 
                             agent.state.energy <= 0 
                             for agent in sim.agents)
                if all_done:
                    break
            
            # Calculate episode metrics
            goal_completion_rate = goals_achieved / (len(sim.agents) * episode_steps)
            resource_stats = {}
            for agent in sim.agents:
                for resource_type, count in agent.state.inventory.items():
                    if resource_type.name not in resource_stats:
                        resource_stats[resource_type.name] = 0
                    resource_stats[resource_type.name] += count
            
            agent_stats = []
            for agent in sim.agents:
                agent_stats.append({
                    "health": agent.state.health,
                    "energy": agent.state.energy,
                    "stress": agent.state.stress,
                    "experience": agent.state.experience,
                    "goal_achieved": agent.goal_achieved,
                    "inventory": {rt.name: count for rt, count in agent.state.inventory.items()}
                })
            
            # Calculate mean rewards for the episode
            mean_step_rewards = [sum(step_rewards) / len(step_rewards) 
                               for step_rewards in episode_rewards]
            
            metrics.add_episode_metrics(
                rewards=mean_step_rewards,
                episode_length=episode_steps,
                goal_completion=goal_completion_rate,
                resource_stats=resource_stats,
                agent_stats=agent_stats
            )
            
            # Print episode summary
            print(f"\nEpisode {episode + 1} Summary:")
            print(f"Steps: {episode_steps}")
            print(f"Average Reward: {np.mean(mean_step_rewards):.2f}")
            print(f"Goal Completion Rate: {goal_completion_rate:.2%}")
            print("Resources Collected:")
            for resource_type, count in resource_stats.items():
                print(f"  {resource_type}: {count}")
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nError during simulation: {e}")
        raise
    finally:
        # Save metrics
        try:
            metrics_file = metrics.save_metrics(output_dir)
            print(f"\nMetrics saved to: {metrics_file}")
        except Exception as e:
            print(f"Error saving metrics: {e}")
        
        # Cleanup
        plt.ioff()
        plt.close('all')
        
        print("\nSimulation completed")

def normalize_experience(experience: float) -> float:
    """Normalize experience value using logarithmic scaling."""
    if experience <= 0:
        return 0.0
    # Use log10 to compress the scale, multiply by 20 to get a good range
    return min(100.0, 20.0 * np.log10(1 + experience))

def print_agent_stats(agents: List[AdaptiveAgent], step: int, max_steps: int):
    """Print formatted agent statistics with relative analysis."""
    analytics = AgentAnalytics.analyze_agents(agents)
    
    print(f"\nStep {step + 1}/{max_steps}")
    print("\n=== Group Statistics ===")
    print(f"Health: {analytics['group_health']['mean']:.1f} ± {analytics['group_health']['std']:.1f}")
    print(f"Energy: {analytics['group_energy']['mean']:.1f} ± {analytics['group_energy']['std']:.1f}")
    print(f"Critical Health Count: {analytics['group_health']['critical_count']}")
    print(f"Critical Energy Count: {analytics['group_energy']['critical_count']}")
    print(f"High Stress Count: {analytics['group_stress']['high_stress_count']}")
    
    print("\n=== Resource Distribution ===")
    for resource, stats in analytics['resource_distribution'].items():
        print(f"{resource}: {stats['total']} total, {stats['mean_per_agent']:.1f} per agent")
    
    print("\n=== Goal Distribution ===")
    for goal, count in analytics['goal_distribution'].items():
        print(f"{goal}: {count} agents")
    
    print("\n=== Individual Agents ===")
    for i, agent in enumerate(agents):
        # Normalize experience to 0-100 scale
        normalized_exp = normalize_experience(agent.state.experience)
        
        print(f"\nAgent {i}:")
        print(f"  Position: ({agent.position.x}, {agent.position.y})")
        print(f"  Health: {agent.state.health:.1f}")
        print(f"  Energy: {agent.state.energy:.1f}")
        print(f"  Stress: {agent.state.stress:.1f}")
        print(f"  Experience: {normalized_exp:.1f}%")  # Show as percentage
        print(f"  Current Goal: {agent.current_goal[1] if agent.current_goal else 'None'}")
        if len(agent.state.inventory) > 0:
            print("  Inventory:")
            for resource_type, count in agent.state.inventory.items():
                if count > 0:
                    print(f"    {resource_type.name}: {count}")

def get_agent_display_info(agent: AdaptiveAgent) -> dict:
    """Get agent display information including birth/death status."""
    # Check if this is a new agent (based on experience being very low)
    is_new = agent.state.experience < 1.0
    # Check if agent is dying/dead
    is_dying = not agent.state.is_alive or agent.state.health <= 0 or agent.state.energy <= 0
    
    return {
        'is_new': is_new,
        'is_dying': is_dying
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run grid world simulation")
    parser.add_argument("--world-size", type=int, default=20,
                      help="Size of the grid world")
    parser.add_argument("--num-agents", type=int, default=5,
                      help="Number of agents in simulation")
    parser.add_argument("--num-episodes", type=int, default=100,
                      help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=100,
                      help="Maximum steps per episode")
    parser.add_argument("--save-interval", type=int, default=10,
                      help="Interval for saving visualization")
    parser.add_argument("--vis-update-interval", type=int, default=1,
                      help="Interval for updating visualization")
    parser.add_argument("--output-dir", type=str, default="simulation_output",
                      help="Directory for saving outputs")
    
    args = parser.parse_args()
    try:
        run_simulation(
            world_size=args.world_size,
            num_agents=args.num_agents,
            num_episodes=args.num_episodes,
            max_steps_per_episode=args.max_steps,
            save_interval=args.save_interval,
            vis_update_interval=args.vis_update_interval,
            output_dir=args.output_dir
        )
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
