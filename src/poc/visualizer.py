import matplotlib
# Use macosx backend for native macOS support
matplotlib.use('macosx')

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from grid_world import GridWorld, AdaptiveAgent, ResourceType, Position

class SimulationVisualizer:
    def __init__(self, world_size: int):
        self.world_size = world_size
        self.fig, (self.grid_ax, self.stats_ax) = plt.subplots(1, 2, figsize=(15, 6))
        self.fig.suptitle('Grid World Simulation')
        
        # Resource colors
        self.resource_colors = {
            ResourceType.FOOD: 'green',
            ResourceType.WATER: 'blue',
            ResourceType.TOOL: 'orange',
            ResourceType.INFORMATION: 'purple'
        }
        
        # Initialize plots
        self.grid_ax.set_title('Grid World State')
        self.grid_ax.set_xlim(-0.5, world_size - 0.5)
        self.grid_ax.set_ylim(-0.5, world_size - 0.5)
        self.grid_ax.grid(True)
        
        self.stats_ax.set_title('Agent Statistics')
        self.stats_ax.set_xlim(0, 1)
        self.stats_ax.set_ylim(0, 100)
    
    def update(self, world: GridWorld, agents: List[AdaptiveAgent], rewards: List[float], step: int):
        """Update visualization with current simulation state."""
        # Clear previous plots
        self.grid_ax.clear()
        self.stats_ax.clear()
        
        # Reset grid properties
        self.grid_ax.set_title('Grid World State')
        self.grid_ax.set_xlim(-0.5, self.world_size - 0.5)
        self.grid_ax.set_ylim(-0.5, self.world_size - 0.5)
        self.grid_ax.grid(True)
        
        # Plot resources
        for pos, resource in world.resources.items():
            color = self.resource_colors[resource.type]
            size = 50 + 20 * (resource.quantity / resource.max_quantity)  # Size varies with quantity
            alpha = 0.5 + 0.5 * (resource.quantity / resource.max_quantity)  # Transparency varies with quantity
            self.grid_ax.scatter(pos.x, pos.y, c=color, s=size, alpha=alpha, marker='s')
        
        # Plot obstacles
        for pos in world.obstacles:
            self.grid_ax.add_patch(plt.Rectangle((pos.x - 0.5, pos.y - 0.5), 1, 1, 
                                               facecolor='gray', alpha=0.5))
        
        # Plot agents
        for i, agent in enumerate(agents):
            self.grid_ax.scatter(agent.position.x, agent.position.y, 
                               c='red', marker='o', s=100, label=f'Agent {i}')
            # Add agent ID label
            self.grid_ax.annotate(f'E:{agent.state.energy:.0f}', 
                                (agent.position.x, agent.position.y),
                                xytext=(0, -10), textcoords='offset points',
                                ha='center', va='top', fontsize=8)
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', 
                      markerfacecolor=color, markersize=10, label=rt.name)
            for rt, color in self.resource_colors.items()
        ]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor='red', markersize=10, 
                                        label='Agent'))
        self.grid_ax.legend(handles=legend_elements, loc='upper right')
        
        # Plot agent statistics
        agent_ids = [f'Agent {i}' for i in range(len(agents))]
        x = np.arange(len(agents))
        width = 0.15
        
        # Plot health, energy, stress, experience, and reward bars
        metrics = {
            'Health': ([agent.state.health for agent in agents], 'red'),
            'Energy': ([agent.state.energy for agent in agents], 'blue'),
            'Stress': ([agent.state.stress for agent in agents], 'yellow'),
            'Experience': ([agent.state.experience for agent in agents], 'green'),
            'Reward': ([reward for reward in rewards], 'purple')
        }
        
        for i, (metric, (values, color)) in enumerate(metrics.items()):
            self.stats_ax.bar(x + i * width, values, width, label=metric, color=color)
        
        self.stats_ax.set_title(f'Agent Statistics (Step {step})')
        self.stats_ax.set_xticks(x + width * 2)
        self.stats_ax.set_xticklabels(agent_ids)
        self.stats_ax.legend()
        
        # Adjust layout and draw
        plt.tight_layout()
        self.fig.canvas.draw()
    
    def save(self, filename: str):
        """Save current visualization to file."""
        self.fig.savefig(filename)
    
    def close(self):
        """Close the visualization."""
        plt.close(self.fig) 
