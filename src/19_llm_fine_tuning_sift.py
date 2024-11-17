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

# Configure logging
logging.basicConfig(level=logging.ERROR, format="%(message)s")
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
        batch_size=32,
        dataset_name="openwebtext",
        split="train",
        streaming=True,
    )
    sampler = SubsetSampler(data_loader)
    metrics = MetricsComputer()
    visualizer = SIFTVisualizer()

    # Initialize trainer
    trainer = SIFTTrainer(
        llm_name="unsloth/Llama-3.2-1B",
        embedding_model="BAAI/bge-large-en-v1.5",
        index_dir="cache/faiss",
        max_length=512,
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
            return "First prompt starting...\n" + "─" * 60
        return (
            "Last Prompt ({}/100) Results:\n"
            "• Avg Loss: {:.4f}\n"
            "• Best Loss: {:.4f}\n"
            "• Global Best: {:.4f}\n"
            "• Examples: {}\n"
            "{}"
        ).format(
            last_stats["prompt_idx"],
            last_stats["avg_loss"],
            last_stats["prompt_best"],
            last_stats["global_best"],
            last_stats["n_examples"],
            "─" * 60,
        )

    def format_header(prompt_idx: int, prompt: str) -> str:
        return ("Prompt {}/100\n" "{}{}\n" "").format(
            prompt_idx + 1,
            prompt[:60].strip().replace("\n", ""),
            "..." if len(prompt) > 60 else "",
        )

    def format_step(step: int, metrics: dict) -> str:
        return (
            f"Step {step:02d} │ "
            f"Loss: {metrics['loss']:.4f} │ "
            f"Best: {metrics['global_best']:.4f} │ "
            f"Δ: {metrics['uncertainty']:.4f}"
        )

    def format_summary(stats: dict) -> str:
        pass

    # Initialize tracking lists
    prompt_stats_history = []

    # Training loop
    last_prompt_stats = None

    for prompt_idx, prompt in enumerate(
        tqdm(test_prompts, desc="Processing prompts", ncols=80, leave=False)
    ):
        try:
            # Disable all other loggers
            logging.getLogger("sift.sift_trainer").setLevel(logging.WARNING)
            logging.getLogger("tqdm").setLevel(logging.WARNING)

            # Print initial header
            logger.info(format_last_prompt_summary(last_prompt_stats))
            logger.info(format_header(prompt_idx, prompt))

            selected_examples = trainer.select_examples(prompt, training_data)
            if not selected_examples:
                continue

            clear_screen()

            # Re-enable logging and reprint header
            logging.getLogger("sift.sift_trainer").setLevel(logging.INFO)

            # Reprint header after clear
            logger.info(format_last_prompt_summary(last_prompt_stats))
            logger.info(format_header(prompt_idx, prompt))

            prompt_stats = {"losses": [], "prompt_best": float("inf")}

            for i, example in enumerate(selected_examples):
                try:
                    step_metrics = trainer.fine_tune_step(example)
                    if step_metrics is None:
                        continue

                    current_loss = min(
                        step_metrics.get("loss", float("inf")), max_loss_threshold
                    )
                    prompt_stats["losses"].append(current_loss)
                    prompt_stats["prompt_best"] = min(
                        prompt_stats["prompt_best"], current_loss
                    )

                    if current_loss < global_best_loss:
                        global_best_loss = current_loss
                        logger.info(f"★ New Best: {global_best_loss:.4f}")

                    # Only log every few steps
                    if i % 3 == 0:
                        logger.info(
                            format_step(
                                i,
                                {
                                    "loss": current_loss,
                                    "global_best": global_best_loss,
                                    "uncertainty": step_metrics.get("uncertainty", 0),
                                },
                            )
                        )

                except Exception as e:
                    continue

            # Store and log summary
            prompt_stats_history.append(prompt_stats)
            if prompt_stats["losses"]:
                logger.info(
                    format_summary(
                        {
                            "avg_loss": sum(prompt_stats["losses"])
                            / len(prompt_stats["losses"]),
                            "prompt_best": prompt_stats["prompt_best"],
                            "global_best": global_best_loss,
                        }
                    )
                )

            # Store summary for next prompt
            if prompt_stats["losses"]:
                last_prompt_stats = {
                    "prompt_idx": prompt_idx + 1,
                    "avg_loss": sum(prompt_stats["losses"])
                    / len(prompt_stats["losses"]),
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


if __name__ == "__main__":
    main()
