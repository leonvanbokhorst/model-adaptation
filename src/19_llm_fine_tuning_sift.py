import logging
from pathlib import Path
from tqdm import tqdm
from sift.sift_data_loading import TextDataLoader, SubsetSampler
from sift.sift_metrics import MetricsComputer
from sift.sift_trainer import SIFTTrainer
from sift.sift_visualization import SIFTVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
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
    
    logger.info(f"Sampled {len(test_prompts)} test prompts and {len(training_data)} training examples")

    # Training loop
    for prompt_idx, prompt in enumerate(tqdm(test_prompts, desc="Processing prompts")):
        try:
            trainer.reset_uncertainty()
            selected_examples = trainer.select_examples(prompt, training_data)
            
            if not selected_examples:
                continue

            for i, example in enumerate(selected_examples):
                try:
                    # Get training metrics
                    step_metrics = trainer.fine_tune_step(example)
                    
                    if step_metrics is None:
                        logger.warning(f"Skipping example {i} due to error")
                        continue
                    
                    # Update metrics history
                    metrics.update(step_metrics)
                    
                    # Get current uncertainty
                    uncertainty = step_metrics.get('uncertainty', float('inf'))
                    
                    # Log progress
                    if i % 10 == 0:
                        logger.info(f"Step {i} - Loss: {step_metrics.get('loss', 'N/A'):.4f}, "
                                  f"Uncertainty: {uncertainty:.4f}")

                except Exception as e:
                    logger.error(f"Error in fine-tuning loop: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error processing prompt {prompt_idx + 1}: {str(e)}")
            continue

    # Visualize results
    metrics_summary = metrics.get_metrics_summary()
    visualizer.plot_metrics_over_time(
        metrics_summary,
        title="Training Progress",
        save_path="metrics.png"
    )


if __name__ == "__main__":
    main()
