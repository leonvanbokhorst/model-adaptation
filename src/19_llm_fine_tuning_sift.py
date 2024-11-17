import logging
import os
import platform
import time
from pathlib import Path
from tqdm import tqdm
from sift.sift_data_loading import TextDataLoader, SubsetSampler
from sift.sift_metrics import MetricsComputer
from sift.sift_trainer import SIFTTrainer
from sift.sift_visualization import SIFTVisualizer
import sys
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    # Configure single handler for all loggers
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]  # Only one handler
    root_logger.setLevel(logging.INFO)

    # Configure our logger - no additional handlers needed
    logger = logging.getLogger(__name__)
    logger.handlers = []  # Remove any existing handlers
    logger.propagate = True  # Use root logger's handler

    # Remove handlers from other loggers and set propagate
    for log_name in ["sift.sift_trainer", "tqdm"]:
        other_logger = logging.getLogger(log_name)
        other_logger.handlers = []
        other_logger.propagate = False  # Don't propagate to root

    # Initialize components
    data_loader = TextDataLoader(
        tokenizer_name="unsloth/Llama-3.2-1B",
        max_length=512,
        batch_size=16,
        dataset_name="openwebtext",
        split="train",
        streaming=True,
    )
    sampler = SubsetSampler(data_loader)
    metrics = MetricsComputer()
    visualizer = SIFTVisualizer()

    # Initialize trainer with enhanced parameters
    trainer = SIFTTrainer(
        llm_name="unsloth/Llama-3.2-1B",
        embedding_model="BAAI/bge-large-en-v1.5",
        index_dir="cache/faiss",
    )

    # Sample test prompts and training data
    n_test_prompts = 100
    test_prompts = sampler.sample_test_prompts(n_prompts=n_test_prompts)
    training_data = sampler.get_training_subset(size=1000)

    logger.info(
        f"Sampled {len(test_prompts)} test prompts and {len(training_data)} training examples"
    )

    # Initialize tracking variables
    best_loss = float("inf")
    global_best_loss = float("inf")
    patience = 15
    patience_counter = 0
    min_examples = 5
    window_size = 5

    # Adjust thresholds for better stability
    initial_delta = 0.2  # Reduced from 0.3
    min_delta = initial_delta
    delta_decay = 0.98  # Slower decay
    min_delta_floor = 0.05  # Lower floor

    # Add stability parameters
    max_loss_threshold = 4.0  # Cap maximum loss
    min_uncertainty = 0.05
    max_uncertainty = 0.8
    moving_window = 3  # Shorter window for quicker reactions

    # Add loss history
    global_losses = []
    prompt_best_losses = []

    # Add log formatting helpers
    def clear_screen():
        # time.sleep(2.1)  # Add small delay before clearing
        if platform.system() == "Windows":
            os.system("cls")
        else:
            os.system("clear")

    def format_last_prompt_summary(last_stats: dict) -> str:
        if not last_stats:
            last_stats = {
                "prompt_idx": 0,
                "avg_loss": 0,
                "prompt_best": 0,
                "global_best": 0,
            }
        return (
            "Results for Prompt ({}/100):\n"
            "Avg Loss:     {:<8.4f}\n"
            "Best Loss:    {:<8.4f}\n"
            "Global Best:  {:<8.4f}\n"
            "{}"
        ).format(
            last_stats["prompt_idx"],
            last_stats["avg_loss"],
            last_stats["prompt_best"],
            last_stats["global_best"],
            "─" * 70,
        )

    def format_header(prompt_idx: int, prompt: str) -> str:
        MAX_LENGTH = 67
        truncated_prompt = prompt[:MAX_LENGTH].strip().replace("\n", " ")
        ellipsis = "..." if len(prompt) > MAX_LENGTH else ""
        return f"\n{truncated_prompt}{ellipsis}\n"

    def format_step(step: int, metrics: dict) -> str:
        return (
            f"Step {step:02d} │ "
            f"Loss: {metrics['loss']:.4f} │ "
            f"Δ: {metrics['uncertainty']:.4f}"
        )

    # Initialize tracking lists
    prompt_stats_history = []

    # Training loop
    last_prompt_stats = None

    # Add these near the start of the main() function after initializing trainer
    metrics_tracker = {
        'global_losses': [],
        'prompt_losses': [],
        'uncertainties': [],
        'steps_per_prompt': []
    }

    for prompt_idx, prompt in enumerate(test_prompts):
        try:
            # Disable all other loggers
            logging.getLogger("sift.sift_trainer").setLevel(logging.WARNING)
            logging.getLogger("tqdm").setLevel(logging.WARNING)

            # Use enhanced selection method
            selected_examples = trainer.select_examples_sift(prompt, training_data)
            logger.info(f"Selected {len(selected_examples)} examples for fine-tuning")
            if not selected_examples:
                logger.warning("No examples selected - skipping prompt")
                continue

            # Re-enable logging and reprint header
            logging.getLogger("sift.sift_trainer").setLevel(logging.INFO)

            #clear_screen()

            # Reprint header after clear
            logger.info(format_last_prompt_summary(last_prompt_stats))
            logger.info(format_header(prompt_idx, prompt))

            prompt_stats = {"losses": [], "prompt_best": float("inf")}

            for i, example in enumerate(selected_examples):
                try:
                    step_metrics = trainer.fine_tune_step(example)
                    if step_metrics is None:
                        continue

                    # Compute kernel-based uncertainty with stability check
                    uncertainties = []
                    for _ in range(3):  # Multiple measurements for stability
                        uncertainty = trainer.compute_kernel_uncertainty(
                            prompt, selected_examples[:i + 1]
                        )
                        if uncertainty is not None and not np.isnan(uncertainty):
                            uncertainties.append(uncertainty)
                            
                    if not uncertainties:
                        continue
                            
                    uncertainty = np.median(uncertainties)  # Use median for robustness
                    current_loss = min(step_metrics.get("loss", float("inf")), max_loss_threshold)
                    
                    # Update tracking
                    metrics_tracker['global_losses'].append(current_loss)
                    metrics_tracker['uncertainties'].append(uncertainty)
                    
                    prompt_stats["losses"].append(current_loss)
                    prompt_stats["prompt_best"] = min(prompt_stats["prompt_best"], current_loss)

                    if current_loss < global_best_loss:
                        global_best_loss = current_loss
                        # Save best model checkpoint
                        trainer.save_checkpoint(f"checkpoints/best_model_{prompt_idx}")

                    # Log progress
                    if i % 1 == 0:
                        logger.info(
                            format_step(
                                i,
                                {
                                    "loss": current_loss,
                                    "prev_loss": prompt_stats["losses"][-2] if len(prompt_stats["losses"]) > 1 else None,
                                    "global_best": global_best_loss,
                                    "uncertainty": uncertainty,
                                },
                            )
                        )

                    # Enhanced stopping check with stability verification
                    if (i >= min_examples and 
                        trainer.should_stop_adaptive(uncertainty, i, alpha=0.1) and
                        len(prompt_stats["losses"]) >= 3 and
                        np.std(prompt_stats["losses"][-3:]) < 0.1):
                        logger.info(f"Stopping early at step {i} due to convergence")
                        break

                except Exception as e:
                    logger.error(f"Error in training step: {str(e)}")
                    continue

            # Store and log summary
            prompt_stats_history.append(prompt_stats)

            # Store summary for next prompt
            if prompt_stats["losses"]:
                current_avg_loss = sum(prompt_stats["losses"]) / len(
                    prompt_stats["losses"]
                )
                last_prompt_stats = {
                    "prompt_idx": prompt_idx + 1,
                    "avg_loss": current_avg_loss,
                    "prev_avg_loss": (
                        last_prompt_stats["avg_loss"] if last_prompt_stats else None
                    ),
                    "prompt_best": prompt_stats["prompt_best"],
                    "global_best": global_best_loss,
                    "n_examples": len(prompt_stats["losses"]),
                }

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            continue

    # Final training summary
    logger.info("\n" + "=" * 35 + " TRAINING COMPLETE " + "=" * 35)

    if prompt_stats_history:  # Add check to prevent division by zero
        avg_examples = sum(len(p["losses"]) for p in prompt_stats_history) / len(
            prompt_stats_history
        )
        total_steps = sum(len(p["losses"]) for p in prompt_stats_history)

        logger.info(
            f"Total Prompts Processed: {len(prompt_stats_history)}\n"
            f"Final Global Best Loss: {global_best_loss:.4f}\n"
            f"Average Examples per Prompt: {avg_examples:.1f}\n"
            f"Total Training Steps: {total_steps}"
        )
    else:
        logger.warning("⚠️ No training statistics collected!")

    logger.info("=" * 89)

    # After training loop, add visualization
    if prompt_stats_history:
        metrics_data = {
            "loss": [stat["losses"] for stat in prompt_stats_history],
            "uncertainty": [
                stat.get("uncertainties", []) for stat in prompt_stats_history
            ],
        }

        visualizer.plot_metrics_over_time(metrics_data)
        visualizer.plot_uncertainty_vs_performance(
            uncertainty=[
                stat.get("uncertainties", [])[-1]
                for stat in prompt_stats_history
                if stat.get("uncertainties")
            ],
            performance=[stat["prompt_best"] for stat in prompt_stats_history],
            save_path="uncertainty_vs_performance.png",
        )
        visualizer.plot_adaptive_stopping(
            metrics={
                "uncertainty": [
                    u
                    for stat in prompt_stats_history
                    for u in stat.get("uncertainties", [])
                ],
                "compute": list(
                    range(
                        sum(
                            len(stat.get("uncertainties", []))
                            for stat in prompt_stats_history
                        )
                    )
                ),
            },
            alpha=0.1,
            save_path="adaptive_stopping.png",
        )


if __name__ == "__main__":
    main()
