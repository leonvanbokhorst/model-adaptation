from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Any, Optional
import asyncio
from abc import ABC, abstractmethod
from ollama import AsyncClient
import json
import pandas as pd
from pathlib import Path
import random
import requests
import zipfile
import io
import shutil
import tarfile
from tqdm import tqdm


class BenchmarkType(Enum):
    MMLU = auto()
    GLUE = auto()
    SUPERGLUE = auto()
    HUMANEVAL = auto()
    GSM8K = auto()
    BIGBENCH = auto()
    HELM = auto()


@dataclass
class BenchmarkResult:
    score: float
    metadata: Dict[str, Any]
    strengths: List[str]
    weaknesses: List[str]


@dataclass
class MMluQuestion:
    question: str
    choices: List[str]
    correct_answer: str
    subject: str


class BenchmarkTask(ABC):
    @abstractmethod
    async def evaluate(self, response: str) -> BenchmarkResult:
        pass

    @abstractmethod
    async def generate_prompt(self) -> str:
        pass


class MMluTask(BenchmarkTask):
    MMLU_URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

    @classmethod
    async def download_dataset(cls, target_dir: str = "data/mmlu") -> None:
        """Download and extract the MMLU dataset."""
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)

        print("Downloading MMLU dataset...")
        try:
            response = requests.get(cls.MMLU_URL, stream=True)
            response.raise_for_status()

            # Get total file size for progress bar
            total_size = int(response.headers.get("content-length", 0))

            # Download with progress bar
            chunks = []
            with tqdm(
                total=total_size, unit="iB", unit_scale=True, desc="Downloading"
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    chunks.append(data)
                    pbar.update(len(data))

            content = b"".join(chunks)

            # Extract with progress bar
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:*") as tar:
                members = [m for m in tar.getmembers() if m.name.endswith("_test.csv")]
                for member in tqdm(members, desc="Extracting files"):
                    member.name = Path(member.name).name
                    tar.extract(member, path=target_path)

            print(f"\nDataset downloaded and extracted to {target_path}")

        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise

    def __init__(self, dataset_path: str = "data/mmlu", download: bool = True):
        """Initialize the MMLU task."""
        self.dataset_path = Path(dataset_path)
        self.questions = None  # Will be loaded after download
        self.download_needed = download and not self.dataset_path.exists()

    async def initialize(self) -> None:
        """Async initialization method to handle dataset download and loading."""
        if self.download_needed:
            await self.download_dataset(str(self.dataset_path))
        self.questions = self._load_questions()

    def _load_questions(self) -> List[MMluQuestion]:
        """Load MMLU questions from the default dataset."""
        # Example subjects to test - can be expanded
        subjects = ["abstract_algebra", "anatomy", "astronomy", "business_ethics"]
        questions = []

        for subject in tqdm(subjects, desc="Loading subjects"):
            try:
                df = pd.read_csv(self.dataset_path / f"{subject}_test.csv", header=None)
                for _, row in df.iterrows():
                    questions.append(
                        MMluQuestion(
                            question=row[0],
                            choices=[row[1], row[2], row[3], row[4]],
                            correct_answer=row[5],
                            subject=subject,
                        )
                    )
            except Exception as e:
                logging.warning(f"Could not load subject {subject}: {e}")

        return questions

    async def generate_prompt(self) -> str:
        """Generate a prompt for a random MMLU question."""
        if not self.questions:
            raise ValueError("No MMLU questions loaded")

        self.current_question = random.choice(self.questions)

        prompt = (
            "Please answer the following multiple choice question by responding with "
            "ONLY the letter (A, B, C, or D) of the correct answer.\n\n"
            f"Question: {self.current_question.question}\n\n"
            "Options:\n"
        )
        for i, choice in enumerate(["A", "B", "C", "D"]):
            prompt += f"{choice}. {self.current_question.choices[i]}\n"
        prompt += "\nYour answer (respond with only A, B, C, or D):"
        return prompt

    async def evaluate(self, response: str) -> BenchmarkResult:
        """Evaluate the model's response for MMLU questions."""
        if not hasattr(self, "current_question"):
            raise ValueError("No current question set. Call generate_prompt first.")

        # Extract the letter answer and handle invalid responses
        response = response.strip().upper()
        valid_answers = {"A", "B", "C", "D"}

        # Find first valid answer in response
        answer = None
        for char in response:
            if char in valid_answers:
                answer = char
                break

        if not answer:
            return BenchmarkResult(
                score=0.0,
                metadata={
                    "subject": self.current_question.subject,
                    "question": self.current_question.question,
                    "model_answer": response,
                    "correct_answer": self.current_question.correct_answer,
                    "error": "Invalid response format",
                },
                strengths=[],
                weaknesses=[
                    f"Invalid response format. Expected A, B, C, or D, got: {response}"
                ],
            )

        correct = answer == self.current_question.correct_answer.upper()

        return BenchmarkResult(
            score=1.0 if correct else 0.0,
            metadata={
                "subject": self.current_question.subject,
                "question": self.current_question.question,
                "model_answer": answer,
                "correct_answer": self.current_question.correct_answer,
                "full_response": response,
            },
            strengths=["Correct answer"] if correct else [],
            weaknesses=(
                []
                if correct
                else [
                    f"Incorrect answer: expected {self.current_question.correct_answer}, got {answer}"
                ]
            ),
        )


class LLMBenchmarkSuite:
    def __init__(
        self,
        model_name: str = "llama2:latest",
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 1,
        questions_per_task: int = 10,
    ):
        self.model_name = model_name
        self.questions_per_task = questions_per_task
        self.tasks: Dict[BenchmarkType, List[BenchmarkTask]] = {}
        self.results: Dict[BenchmarkType, List[BenchmarkResult]] = {}
        self.client = AsyncClient()
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    async def _get_llm_response(self, prompt: str) -> str:
        """Get response from Ollama LLM with deterministic settings.

        Args:
            prompt: Input prompt for the model

        Returns:
            Model's response as string
        """
        try:
            response = await self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                },
            )
            return response["message"]["content"]
        except Exception as e:
            logging.error(f"Error getting LLM response: {e}")
            raise

    async def add_benchmark(
        self, benchmark_type: BenchmarkType, tasks: List[BenchmarkTask]
    ):
        """Add benchmark tasks to the suite.

        Args:
            benchmark_type: Type of benchmark to add
            tasks: List of benchmark tasks to execute
        """
        self.tasks[benchmark_type] = tasks

    async def run_benchmarks(
        self, selected_types: Optional[List[BenchmarkType]] = None
    ) -> Dict[str, float]:
        """Run selected benchmarks and return aggregated scores."""
        types_to_run = selected_types or list(self.tasks.keys())
        scores = {}

        for benchmark_type in tqdm(types_to_run, desc="Running benchmarks"):
            if benchmark_type not in self.tasks:
                continue

            results = []
            tasks = self.tasks[benchmark_type]
            for task in tqdm(
                tasks, desc=f"Running {benchmark_type.name} tasks", leave=False
            ):
                # Run multiple questions per task
                for _ in tqdm(
                    range(self.questions_per_task), desc="Questions", leave=False
                ):
                    prompt = await task.generate_prompt()
                    response = await self._get_llm_response(prompt)
                    result = await task.evaluate(response)
                    results.append(result)

            self.results[benchmark_type] = results
            scores[benchmark_type.name] = sum(r.score for r in results) / len(results)

        return scores

    async def generate_report(self) -> str:
        """Generate a detailed report of benchmark results."""
        report = []

        for benchmark_type, results in tqdm(
            self.results.items(), desc="Generating report"
        ):
            report.append(f"\n=== {benchmark_type.name} Results ===")
            total_score = 0

            for i, result in enumerate(results, 1):
                report.append(f"\nTask {i}:")
                report.append(f"Score: {result.score:.2f}")
                if result.metadata:
                    report.append("Metadata:")
                    for k, v in result.metadata.items():
                        report.append(f"  {k}: {v}")
                if result.strengths:
                    report.append("Strengths:")
                    for s in result.strengths:
                        report.append(f"  + {s}")
                if result.weaknesses:
                    report.append("Weaknesses:")
                    for w in result.weaknesses:
                        report.append(f"  - {w}")
                total_score += result.score

            avg_score = total_score / len(results)
            report.append(f"\nAverage Score: {avg_score:.2f}")

        return "\n".join(report)


# Example usage:
async def main():

    suite = LLMBenchmarkSuite(model_name="mistral-nemo:latest", questions_per_task=100)

    with tqdm(total=4, desc="Benchmark Setup") as pbar:
        # Initialize MMLU
        mmlu_task = MMluTask(download=True)
        await mmlu_task.initialize()
        pbar.update(1)

        # Add benchmarks
        await suite.add_benchmark(BenchmarkType.MMLU, [mmlu_task])
        pbar.update(1)

        # Run benchmarks
        scores = await suite.run_benchmarks([BenchmarkType.MMLU])
        pbar.update(1)

        # Generate report
        report = await suite.generate_report()
        pbar.update(1)

    print("\nBenchmark Results:")
    print(json.dumps(scores, indent=2))
    print("\nDetailed Report:")
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
