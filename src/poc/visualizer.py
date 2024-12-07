import matplotlib
# Use macosx backend for native macOS support
matplotlib.use('macosx')

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from grid_world import GridWorld, AdaptiveAgent, ResourceType, Position

class SimulationVisualizer:
    def __init__(self, world_size: int = 10):
        self.world_size = world_size
        
        # Create figure and subplots with native macOS support
        self.fig, (self.world_ax, self.stats_ax) = plt.subplots(1, 2, figsize=(15, 6))
        self.fig.canvas.manager.set_window_title('Grid World Simulation')
        
        # Resource colors for visualization
        self.resource_colors = {
            ResourceType.FOOD: 'green',
            ResourceType.WATER: 'blue',
            ResourceType.TOOL: 'orange',
            ResourceType.INFORMATION: 'purple'
        }
        
        # Initialize plots
        self.world_ax.set_title('Grid World State')
        self.stats_ax.set_title('Agent Statistics')
        
        # Show the plot
        plt.show(block=False)
        self.fig.canvas.draw()
        
    def update(self, 
               world: GridWorld, 
               agents: List[AdaptiveAgent], 
               rewards: List[float],
               step: int):
        """Update visualization with current simulation state."""
        try:
            self._plot_world(world, agents)
            self._plot_stats(agents, rewards, step)
            
            # Adjust layout and redraw
            self.fig.tight_layout()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception as e:
            print(f"Visualization update error: {e}")
        
    def _plot_world(self, world: GridWorld, agents: List[AdaptiveAgent]):
        """Plot the grid world state."""
        self.world_ax.clear()
        self.world_ax.set_title('Grid World State')
        
        # Plot grid
        self.world_ax.set_xlim(-0.5, world.size - 0.5)
        self.world_ax.set_ylim(-0.5, world.size - 0.5)
        self.world_ax.grid(True)
        
        # Plot obstacles
        for obstacle in world.obstacles:
            self.world_ax.add_patch(plt.Rectangle(
                (obstacle.x - 0.5, obstacle.y - 0.5),
                1, 1, color='gray', alpha=0.5
            ))
            
        # Plot resources
        for pos, resource_type in world.resources.items():
            self.world_ax.scatter(
                pos.x, pos.y,
                color=self.resource_colors[resource_type],
                marker='s', s=100, label=resource_type.name
            )
            
        # Plot agents
        for i, agent in enumerate(agents):
            # Plot agent position
            self.world_ax.scatter(
                agent.position.x, agent.position.y,
                color='red', marker='o', s=150,
                label=f'Agent {i}'
            )
            
            # Plot agent stress level as a halo
            stress_circle = plt.Circle(
                (agent.position.x, agent.position.y),
                0.4, color='yellow',
                alpha=agent.state.stress / 100.0
            )
            self.world_ax.add_patch(stress_circle)
            
            # Add energy level as text
            self.world_ax.text(
                agent.position.x, agent.position.y + 0.2,
                f'E:{agent.state.energy:.0f}',
                ha='center', va='bottom'
            )
            
        # Add legend with unique entries
        handles, labels = self.world_ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.world_ax.legend(
            by_label.values(),
            by_label.keys(),
            loc='center left',
            bbox_to_anchor=(1, 0.5)
        )
        
    def _plot_stats(self, 
                    agents: List[AdaptiveAgent], 
                    rewards: List[float],
                    step: int):
        """Plot agent statistics."""
        self.stats_ax.clear()
        self.stats_ax.set_title(f'Agent Statistics (Step {step})')
        
        # Prepare data
        agent_ids = range(len(agents))
        width = 0.15
        
        # Plot health bars
        health_bars = self.stats_ax.bar(
            [x - width*2 for x in agent_ids],
            [agent.state.health for agent in agents],
            width,
            label='Health',
            color='red'
        )
        
        # Plot energy bars
        energy_bars = self.stats_ax.bar(
            [x - width for x in agent_ids],
            [agent.state.energy for agent in agents],
            width,
            label='Energy',
            color='blue'
        )
        
        # Plot stress bars
        stress_bars = self.stats_ax.bar(
            agent_ids,
            [agent.state.stress for agent in agents],
            width,
            label='Stress',
            color='yellow'
        )
        
        # Plot experience bars
        exp_bars = self.stats_ax.bar(
            [x + width for x in agent_ids],
            [agent.state.experience for agent in agents],
            width,
            label='Experience',
            color='green'
        )
        
        # Plot rewards
        reward_bars = self.stats_ax.bar(
            [x + width*2 for x in agent_ids],
            rewards,
            width,
            label='Reward',
            color='purple'
        )
        
        # Customize the plot
        self.stats_ax.set_ylabel('Value')
        self.stats_ax.set_xlabel('Agent ID')
        self.stats_ax.legend()
        self.stats_ax.set_xticks(agent_ids)
        self.stats_ax.set_xticklabels([f'Agent {i}' for i in agent_ids])
        
    def save(self, filename: str):
        """Save the current visualization state to a file."""
        self.fig.savefig(filename, bbox_inches='tight', dpi=300)
        
    def close(self):
        """Close the visualization window."""
        plt.close(self.fig) 