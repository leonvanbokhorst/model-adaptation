import argparse
from grid_world import Simulation
from visualizer import SimulationVisualizer
import time
import os
import matplotlib.pyplot as plt
import sys

def run_simulation(
    world_size: int = 6,
    num_agents: int = 5,
    num_steps: int = 1000,
    save_interval: int = 10,
    vis_update_interval: int = 1,
    output_dir: str = "simulation_output"
):
    """Run the grid world simulation with visualization."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize simulation
    sim = Simulation(world_size=world_size, num_agents=num_agents)
    vis = None
    
    try:
        # Initialize visualizer
        vis = SimulationVisualizer(world_size=world_size)
        
        # Run simulation steps
        for step in range(num_steps):
            print(f"\nStep {step + 1}/{num_steps}")
            
            # Execute simulation step
            rewards = sim.step()
            
            # Update visualization less frequently
            if vis is not None and step % vis_update_interval == 0:
                try:
                    vis.update(sim.world, sim.agents, rewards, step)
                    
                    # Save visualization at intervals
                    if (step + 1) % save_interval == 0:
                        vis.save(os.path.join(output_dir, f"step_{step + 1}.png"))
                except Exception as e:
                    print(f"Visualization error: {e}")
                    vis = None  # Disable visualization on error
                
            # Print agent stats less frequently
            if step % vis_update_interval == 0:
                print("\nAgent Statistics:")
                for i, agent in enumerate(sim.agents):
                    print(f"\nAgent {i}:")
                    print(f"  Position: ({agent.position.x}, {agent.position.y})")
                    print(f"  Health: {agent.state.health:.1f}")
                    print(f"  Energy: {agent.state.energy:.1f}")
                    print(f"  Stress: {agent.state.stress:.1f}")
                    print(f"  Experience: {agent.state.experience:.1f}")
                    if len(agent.state.inventory) > 0:  # Only print non-empty inventory
                        print("  Inventory:")
                        for resource, count in agent.state.inventory.items():
                            if count > 0:
                                print(f"    {resource.name}: {count}")
                    print(f"  Reward: {rewards[i]:.1f}")
            
            # Keep GUI responsive if visualization is active
            if vis is not None and step % vis_update_interval == 0:
                plt.pause(0.05)  # Reduced pause time
            else:
                time.sleep(0.01)  # Minimal sleep when not updating
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nError during simulation: {e}")
        raise
    finally:
        # Ensure visualization is properly closed if it was initialized
        if vis is not None:
            try:
                vis.close()
                plt.close('all')
            except Exception as e:
                print(f"Error closing visualization: {e}")
        print("\nSimulation completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run grid world simulation")
    parser.add_argument("--world-size", type=int, default=20,
                      help="Size of the grid world")
    parser.add_argument("--num-agents", type=int, default=5,
                      help="Number of agents in simulation")
    parser.add_argument("--num-steps", type=int, default=1000,
                      help="Number of simulation steps")
    parser.add_argument("--save-interval", type=int, default=10,
                      help="Interval for saving visualization")
    parser.add_argument("--vis-update-interval", type=int, default=1,
                      help="Interval for updating visualization")
    parser.add_argument("--output-dir", type=str, default="simulation_output",
                      help="Directory for saving visualization outputs")
    
    args = parser.parse_args()
    try:
        run_simulation(
            world_size=args.world_size,
            num_agents=args.num_agents,
            num_steps=args.num_steps,
            save_interval=args.save_interval,
            vis_update_interval=args.vis_update_interval,
            output_dir=args.output_dir
        )
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1) 
