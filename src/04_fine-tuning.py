from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    LlamaTokenizer,
    LlamaForCausalLM,
)
from peft import LoraConfig, get_peft_model
import os
import dotenv
import torch

dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


def initialize_tokenizer(model_name: str, hf_token: str) -> AutoTokenizer:
    """Initialize and configure the tokenizer with proper padding tokens.

    Args:
        model_name: Name of the model to load tokenizer for
        hf_token: Hugging Face API token

    Returns:
        AutoTokenizer: Configured tokenizer instance
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        add_eos_token=True,
        add_bos_token=True,
        token=hf_token,
    )

    # Set pad token to eos token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def prepare_fine_tuning():
    """Setup and prepare model for LoRA fine-tuning"""
    model_name = "unsloth/Llama-3.2-1B"  # Smaller model better suited for M3

    if not HF_TOKEN:
        raise ValueError("Please set the HF_TOKEN environment variable")

    # Optimize for M3 chip
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        token=HF_TOKEN, 
        device_map="mps",  # Use Metal Performance Shaders
        torch_dtype=torch.float16,
        use_cache=False  # Disable KV-cache during training
    )

    # Enable gradient computation
    model.train()  # Add this line
    model.enable_input_require_grads()  # Add this line

    tokenizer = initialize_tokenizer(model_name, HF_TOKEN)

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    return model, tokenizer


def prepare_dataset(tokenizer):
    """Prepare dataset for fine-tuning"""
    def tokenize_function(examples):
        """Tokenize the text examples"""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors=None,
        )

    print("Loading dataset...")
    # Updated dataset source
    dataset = load_dataset("leonvanbokhorst/synthetic-complaints")
    
    # Add debug logging
    print("Dataset structure:", dataset["train"].features)
    print("First example:", dataset["train"][0])

    def format_prompt(example):
        """Format each example into Llama instruction format"""
        # Updated to match dataset structure
        return {
            "text": f"[INST] {example['instruction']} [/INST] {example['response']}"
        }

    print("Formatting prompts...")
    formatted_dataset = dataset["train"].map(format_prompt)

    print("Tokenizing dataset...")
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing",
        remove_columns=formatted_dataset.column_names,
    )

    return tokenized_dataset


def train_model(model, tokenizer, dataset):
    """Train the model using optimized parameters for M3"""
    training_args = TrainingArguments(
        output_dir="./lora_finetuned",
        num_train_epochs=1,
        per_device_train_batch_size=1,      # Reduced for M3 memory constraints
        gradient_accumulation_steps=8,      # Increased to compensate for smaller batch size
        save_steps=200,                     # Reduced checkpoint frequency
        logging_steps=50,
        learning_rate=1e-4,                # Slightly reduced learning rate
        weight_decay=0.05,
        fp16=False,                        # Disable mixed precision for MPS
        bf16=False,                        # Disable bfloat16
        optim="adamw_torch",
        lr_scheduler_type="cosine",        # Simplified scheduler
        warmup_ratio=0.05,                 # Add warmup period
        gradient_checkpointing=True,       # Enable gradient checkpointing
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Add this line
        max_grad_norm=0.3,                 # Add gradient clipping
    )

    # Initialize trainer with progress bar
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Starting training...")
    trainer.train()

    return trainer


def inference_example(model, tokenizer, prompt):
    """Generate text using fine-tuned model with M3-optimized settings"""
    formatted_prompt = f"[INST] {prompt} [/INST]"
    
    device = "mps"  # Use Metal Performance Shaders
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_length=512,
        temperature=0.7,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,              # Enable KV-cache
        max_new_tokens=256,          # Limit output length
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Usage example
if __name__ == "__main__":
    FINETUNING = True

    if FINETUNING:
        # Setup
        model, tokenizer = prepare_fine_tuning()

        # Prepare dataset
        tokenized_dataset = prepare_dataset(tokenizer)

        # Train
        trainer = train_model(model, tokenizer, tokenized_dataset)

        # Save
        model.save_pretrained("./lora_finetuned")
        tokenizer.save_pretrained("./lora_finetuned")

        # Example inference after training
        prompt = "Write a frustrated complaint about poor customer service"
    else:
        # Load
        tokenizer = AutoTokenizer.from_pretrained("./lora_finetuned")
        model = AutoModelForCausalLM.from_pretrained("./lora_finetuned")

        # Example inference
        prompt = "Instruction: Generate a lighthearted joke about AI ethics and bias in a Professional setting"

    response = inference_example(model, tokenizer, prompt)
    print(response)
