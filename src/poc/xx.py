import os
import subprocess

# Set tokenizer parallelism to false to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.optim as optim
import time
from sentence_transformers import SentenceTransformer
from typing import Optional, Tuple, List
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)

embed_dim = 32
num_agents = 5

# Laad een sentence transformer voor semantische tekst→embedding mapping
text_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Add this after text_embedder initialization
embedding_reducer = nn.Linear(384, embed_dim)  # Reduce from 384 to embed_dim dimensions
embedding_reducer = embedding_reducer.to(device)


def interpret_embedding_with_llm(
    embedding: torch.Tensor, goal_embedding: torch.Tensor
) -> torch.Tensor:
    """Interprets agent's current state and goal using LLM to generate advice.

    Args:
        embedding: Current state embedding tensor
        goal_embedding: Goal state embedding tensor

    Returns:
        torch.Tensor: Reduced embedding of LLM advice
    """
    emb_list = embedding.squeeze().tolist()
    emb_str = ", ".join([f"{x:.2f}" for x in emb_list])
    goal_list = goal_embedding.squeeze().tolist()
    goal_str = ", ".join([f"{x:.2f}" for x in goal_list])

    prompt = f"""
Denk aan deze vector van innerlijke toestanden: [{emb_str}].
Dit vertegenwoordigt de huidige geestesgesteldheid van een agent.
Deze agent heeft een doel, vertegenwoordigd door deze vector: [{goal_str}].
Beschrijf het innerlijke landschap van de agent en geef specifiek advies:
Hoe kan de agent zijn huidige innerlijke toestand dichter bij het doel brengen?
Focus op emoties, motivaties, en mogelijke stappen om richting het doel te bewegen.
"""

    # Run Ollama lokaal
    try:
        result = subprocess.run(
            ["ollama", "run", "hermes3:latest"],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,  # Add timeout
        )
        result.check_returncode()  # Raise exception on non-zero exit
    except subprocess.TimeoutExpired:
        print("LLM request timed out")
        # Handle timeout gracefully
    except subprocess.CalledProcessError as e:
        print(f"LLM request failed: {e}")
        # Handle error gracefully

    output_text = result.stdout.decode("utf-8").strip()
    # print(f"\n🤖 LLM Advice:\n{output_text}\n")
    # Embed de Ollama-output semantisch
    advice_embedding = text_embedder.encode([output_text], convert_to_tensor=True)
    # Move advice_embedding to device before reduction
    advice_embedding = advice_embedding.to(device)
    reduced_advice = embedding_reducer(advice_embedding)
    return reduced_advice


@dataclass
class EmotionalMemory:
    embedding: torch.Tensor
    intensity: float
    decay_rate: float
    timestamp: int


class EmotionalState(Enum):
    FLOW = auto()
    RESISTANCE = auto()
    GROWTH = auto()
    REGRESSION = auto()


class Agent(nn.Module):
    def __init__(self, embed_dim: int = 32):
        """Initialize agent with internal state vectors.

        Args:
            embed_dim: Dimension of embedding vectors
        """
        super().__init__()

        # Initialize with some random baseline emotions and trauma
        self.personality = nn.Parameter(
            torch.randn(embed_dim) * 0.5, requires_grad=False
        )
        self.trauma = nn.Parameter(
            torch.rand(embed_dim) * 0.1, requires_grad=False
        )  # Start with small random trauma
        self.emotion = nn.Parameter(
            torch.randn(embed_dim) * 0.2, requires_grad=False
        )  # Start with random emotions
        self.politics = nn.Parameter(torch.randn(embed_dim) * 0.1, requires_grad=False)
        self.desires = nn.Parameter(torch.randn(embed_dim) * 0.1, requires_grad=False)
        self.goal = nn.Parameter(torch.randn(embed_dim) * 0.3, requires_grad=False)

        # Increase base adaptation rate for more pronounced changes
        self.adaptation_rate = nn.Parameter(torch.tensor(0.3))  # Increased from 0.1
        self.resilience = nn.Parameter(torch.tensor(0.5))

        # Enhanced internal states
        self.memories: List[EmotionalMemory] = []
        self.state_history = torch.zeros((100, embed_dim))  # Rolling window
        self.attention = nn.MultiheadAttention(embed_dim, 4)
        self.emotional_state = EmotionalState.FLOW

        # Enhanced neural processing
        self.state_processor = nn.Sequential(
            nn.Linear(embed_dim * 6, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.to(device)

    def process_memory(self, current_state: torch.Tensor) -> torch.Tensor:
        """Process emotional memories and their influence on current state."""
        if not self.memories:
            return current_state

        memory_tensor = torch.stack([m.embedding for m in self.memories])
        attn_output, _ = self.attention(
            current_state.unsqueeze(0),
            memory_tensor.unsqueeze(0),
            memory_tensor.unsqueeze(0),
        )

        # Decay and prune old memories
        self.memories = [
            m
            for m in self.memories
            if m.intensity * (m.decay_rate ** (step - m.timestamp)) > 0.1
        ]

        return attn_output.squeeze(0)

    def forward(
        self, external_context: torch.Tensor
    ) -> Tuple[torch.Tensor, EmotionalState]:
        # Move external_context to the correct device if needed
        external_context = external_context.to(device)

        # Update state history
        self.state_history = self.state_history.to(device)
        self.state_history = torch.roll(self.state_history, -1, dims=0)

        # Process current context with memory influence
        memory_influenced = self.process_memory(external_context)

        # Ensure all tensors have the same dimensions and are on the same device
        internal_states = torch.cat(
            [
                self.personality.to(device).unsqueeze(0),
                self.emotion.to(device).unsqueeze(0),
                self.trauma.to(device).unsqueeze(0),
                self.politics.to(device).unsqueeze(0),
                self.desires.to(device).unsqueeze(0),
                (
                    memory_influenced.unsqueeze(0)
                    if memory_influenced.dim() == 1
                    else memory_influenced
                ).to(device),
            ],
            dim=1,
        )

        # Process state with enhanced neural network
        processed_state = self.state_processor(internal_states)

        # Store the processed state in history
        self.state_history[-1] = processed_state.squeeze(0)

        # Determine emotional state
        state_velocity = torch.diff(self.state_history[-10:], dim=0).mean(dim=0)
        goal_distance = torch.norm(processed_state - self.goal.to(device))

        if goal_distance < 0.1:
            new_emotional_state = EmotionalState.FLOW
        elif torch.dot(state_velocity, self.goal.to(device)) > 0:
            new_emotional_state = EmotionalState.GROWTH
        elif goal_distance > torch.norm(self.state_history[-10] - self.goal.to(device)):
            new_emotional_state = EmotionalState.REGRESSION
        else:
            new_emotional_state = EmotionalState.RESISTANCE

        self.emotional_state = new_emotional_state

        return processed_state.squeeze(0), new_emotional_state

    def update_from_advice(self, advice_embedding: torch.Tensor) -> None:
        with torch.no_grad():
            advice_embedding = advice_embedding.to(device)

            # Increase base step size and emotional impact
            base_step = 0.05 * self.adaptation_rate.to(device)  # Increased from 0.01
            emotional_intensity = (
                torch.norm(advice_embedding) * 1.5
            )  # Amplified emotional intensity

            # More pronounced step size adjustments
            if self.emotional_state == EmotionalState.RESISTANCE:
                step_size = base_step * 0.7  # Less reduction
            elif self.emotional_state == EmotionalState.GROWTH:
                step_size = base_step * 2.0  # More growth
            elif self.emotional_state == EmotionalState.FLOW:
                step_size = base_step * 1.5
            else:
                step_size = base_step

            # Stronger emotional updates
            emotion_update = step_size * (
                advice_embedding.squeeze(0) - self.emotion.to(device)
            )
            new_emotion = (
                self.emotion.to(device) + emotion_update * emotional_intensity
            ).clamp(-1, 1)
            self.emotion.data.copy_(new_emotion)

            # More impactful desire updates
            desire_strength = (
                torch.sigmoid(torch.norm(self.desires.to(device))) * 1.5
            )  # Amplified
            desire_update = (
                step_size
                * desire_strength
                * (advice_embedding.squeeze(0) - self.desires.to(device))
            )
            new_desires = (self.desires.to(device) + desire_update).clamp(-1, 1)
            self.desires.data.copy_(new_desires)

            # More sensitive trauma processing
            trauma_healing = self.resilience.to(device) * torch.sigmoid(
                torch.norm(self.trauma.to(device))
            )
            trauma_increase = (
                torch.relu(-emotion_update).mean() * 0.3
            )  # Increased from 0.1
            new_trauma = (
                self.trauma.to(device) * (1 - trauma_healing * 0.02) + trauma_increase
            ).clamp(0, 1)
            self.trauma.data.copy_(new_trauma)

            # Lower threshold for memory storage
            if torch.norm(emotion_update) > 0.05:  # Reduced from 0.1
                self.memories.append(
                    EmotionalMemory(
                        embedding=self.emotion.clone(),
                        intensity=emotional_intensity.item(),
                        decay_rate=0.95 + 0.04 * emotional_intensity.item(),
                        timestamp=step,
                    )
                )


agents = [Agent(embed_dim) for _ in range(num_agents)]


def get_external_context(step):
    angle = torch.tensor(step * 0.01)
    context = torch.zeros(embed_dim)
    context[0] = torch.sin(angle) * 0.5
    context[1] = torch.cos(angle) * 0.5
    context += 0.01 * torch.randn(embed_dim)
    # Move context to device
    return context.unsqueeze(0).to(device)


step = 0
try:
    while True:
        print(f"\n=== Simulation Step {step} ===")
        step += 1
        external_context = get_external_context(step)
        print(f"🌍 External Context Mean: {external_context.mean():.3f}")

        # Voor elke agent:
        outputs = []
        emotional_states = []
        for idx, agent in enumerate(agents):
            current_state, emotional_state = agent(external_context)
            print(f"\n👤 Agent {idx + 1}:")
            print(f"  Current State Mean: {current_state.mean():.3f}")
            print(f"  Goal State Mean: {agent.goal.mean():.3f}")
            print(f"  Emotion Mean: {agent.emotion.mean():.3f}")
            print(f"  Trauma Level: {agent.trauma.abs().mean():.3f}")
            print(f"  Emotional State: {emotional_state.name}")

            advice_embedding = interpret_embedding_with_llm(
                current_state.unsqueeze(0), agent.goal.unsqueeze(0)
            )
            agent.update_from_advice(advice_embedding)
            outputs.append(current_state)
            emotional_states.append(emotional_state)

        outputs = torch.stack(outputs)
        mean_embedding = outputs.mean(dim=0)
        direction_factor = 1.0 if mean_embedding.mean() < 0 else -1.0
        print(f"\n📊 Group Statistics:")
        print(f"  Mean Group State: {mean_embedding.mean():.3f}")
        print(f"  Direction Factor: {direction_factor}")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n⚠️ Simulation stopped by user")
