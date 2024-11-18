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
from datasets import load_dataset
import os
from dotenv import load_dotenv
import logging
import httpx
import re


class BenchmarkType(Enum):
    MMLU = auto()
    GLUE = auto()
    SUPERGLUE = auto()
    HUMANEVAL = auto()
    GSM8K = auto()
    BIGBENCH = auto()
    HELM = auto()
    GPQA = auto()


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


@dataclass
class GPQAQuestion:
    question: str
    choices: List[str]
    correct_answer: str
    subject: str
    metadata: Dict[str, Any]


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


class GPQATask(BenchmarkTask):
    GPQA_URL = "https://huggingface.co/datasets/Idavidrein/gpqa"

    def __init__(
        self,
        dataset_path: str = "data/gpqa",
        use_cached: bool = True,
        domains: List[str] = None,
    ):
        """Initialize GPQA task with specific domains."""
        self.dataset_path = Path(dataset_path)
        self.use_cached = use_cached
        self.domains = domains or [
            # Core Science & Math
            "basic_mathematics",
            "statistics",
            "probability",
            "linear_algebra",
            # Applied Sciences
            "computer_science",
            "data_science",
            "software_engineering",
            "machine_learning",
            # Business & Economics
            "economics",
            "finance",
            "business_strategy",
            "market_analysis",
            # Problem Solving
            "logic_puzzles",
            "optimization",
            "decision_making",
            "system_design",
            # Real-world Applications
            "project_management",
            "risk_analysis",
            "technical_writing",
            "data_analysis",
        ]
        self.questions = None

    async def initialize(self) -> None:
        """Initialize the dataset - requires HuggingFace authentication."""
        load_dotenv()

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "HF_TOKEN environment variable not found. "
                "Please set your HuggingFace token in .env file. "
                "You can get a token from https://huggingface.co/settings/tokens"
            )

        try:
            if self.use_cached and self.dataset_path.exists():
                try:
                    self.questions = self._load_cached_questions()
                    logging.info("Successfully loaded cached GPQA questions")
                    return
                except FileNotFoundError:
                    logging.info(
                        "No cached GPQA questions found, loading from HuggingFace..."
                    )

            os.environ["HUGGINGFACE_TOKEN"] = hf_token

            logging.info(
                f"Loading GPQA dataset with domains: {', '.join(self.domains)}"
            )

            # Load each domain separately and combine
            all_questions = []
            for domain in self.domains:
                try:
                    dataset = load_dataset(
                        "Idavidrein/gpqa",
                        domain,  # Load one domain at a time
                        token=hf_token,  # Use token instead of use_auth_token
                    )
                    if dataset and "train" in dataset:
                        domain_questions = self._convert_to_questions(dataset)
                        all_questions.extend(domain_questions)
                        logging.info(
                            f"Loaded {len(domain_questions)} questions from {domain}"
                        )
                except Exception as e:
                    logging.warning(f"Failed to load domain {domain}: {e}")
                    continue

            if not all_questions:
                raise ValueError("No questions could be loaded from any domain")

            self.questions = all_questions
            self._cache_questions()
            logging.info(f"Successfully loaded {len(self.questions)} GPQA questions")

        except Exception as e:
            logging.error(f"Error loading GPQA dataset: {e}", exc_info=True)
            raise

    def _convert_to_questions(self, dataset) -> List[GPQAQuestion]:
        """Convert HuggingFace dataset to internal format."""
        questions = []

        for item in dataset["train"]:
            try:
                # Extract core question fields
                question = item.get("Question")
                correct_answer = item.get("Correct Answer")
                incorrect_1 = item.get("Incorrect Answer 1")
                incorrect_2 = item.get("Incorrect Answer 2")
                incorrect_3 = item.get("Incorrect Answer 3")

                # Skip if missing core fields
                if not all(
                    [question, correct_answer, incorrect_1, incorrect_2, incorrect_3]
                ):
                    continue

                # Construct choices list
                choices = [correct_answer, incorrect_1, incorrect_2, incorrect_3]
                random.shuffle(choices)  # Randomize order

                # Get metadata
                metadata = {
                    "domain": item.get("High-level domain", "Unknown"),
                    "subdomain": item.get("Subdomain", "Unknown"),
                    "difficulty": item.get("Writer's Difficulty Estimate", "Unknown"),
                    "expert_accuracy": item.get("Expert Validator Accuracy", None),
                    "non_expert_accuracy": item.get(
                        "Non-Expert Validator Accuracy", None
                    ),
                }

                questions.append(
                    GPQAQuestion(
                        question=question,
                        choices=choices,
                        correct_answer=correct_answer,
                        subject=metadata["subdomain"],
                        metadata=metadata,
                    )
                )

            except Exception as e:
                logging.warning(f"Error processing GPQA question: {str(e)}")
                continue

        if not questions:
            raise ValueError("No valid questions could be parsed from the GPQA dataset")

        return questions

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answers for comparison."""
        # Remove whitespace and convert to lowercase
        answer = answer.lower().strip()

        # Handle fractions
        fraction_map = {
            "1/4": "quarter",
            "1/2": "half",
            "3/4": "three quarters",
            "1/3": "third",
            "2/3": "two thirds",
        }
        for frac, word in fraction_map.items():
            if frac in answer:
                answer = answer.replace(frac, word)

        # Handle numeric answers
        try:
            # Extract first number if present
            num_match = re.search(r"[-+]?\d*\.?\d+", answer)
            if num_match:
                num = float(num_match.group())
                # Convert to integer if it's a whole number
                if num.is_integer():
                    return str(int(num))
                return f"{num:.2f}"
        except:
            pass

        # Remove common words and punctuation
        answer = answer.replace("the", "").replace("a", "").replace("an", "")
        answer = re.sub(r"[^\w\s]", "", answer)
        answer = " ".join(answer.split())  # Normalize whitespace

        return answer

    async def generate_prompt(self) -> str:
        """Generate a concise prompt for GPQA questions."""
        if not self.questions:
            raise ValueError("No GPQA questions loaded")

        self.current_question = random.choice(self.questions)
        is_multiple_choice = len(self.current_question.correct_answer.strip()) == 1

        system_prompt = (
            "You are a precise answering system. Give ONLY the answer, no explanations.\n"
            "- For multiple choice: respond with only A, B, C, or D\n"
            "- For numbers: give only the number with units if needed\n"
            "- For text: give only the exact text answer\n"
            "Never explain your reasoning."
        )

        if is_multiple_choice:
            prompt = (
                "RESPOND WITH ONLY A SINGLE LETTER: A, B, C, or D\n\n"
                f"Question: {self.current_question.question}\n\n"
                "Options:\n"
            )
            for i, choice in enumerate(self.current_question.choices):
                prompt += f"{chr(65+i)}. {choice}\n"
        else:
            prompt = (
                "RESPOND WITH ONLY THE ANSWER - NO EXPLANATIONS\n\n"
                f"Question: {self.current_question.question}\n\n"
                "Answer (give only the number or exact text):"
            )

        self.system_prompt = system_prompt
        return prompt

    async def evaluate(self, response: str) -> BenchmarkResult:
        """Evaluate the model's response for GPQA questions."""
        if not hasattr(self, "current_question"):
            raise ValueError("No current question set")

        response = response.strip().split("\n")[0]  # Take only first line
        is_multiple_choice = len(self.current_question.correct_answer.strip()) == 1

        if is_multiple_choice:
            answer = next(
                (char for char in response.upper() if char in {"A", "B", "C", "D"}),
                None,
            )
            correct = answer == self.current_question.correct_answer.upper()
        else:
            # Normalize both answers
            answer = self._normalize_answer(response)
            correct_norm = self._normalize_answer(self.current_question.correct_answer)

            # Check for numeric match first
            try:
                answer_num = float(answer)
                correct_num = float(correct_norm)
                correct = abs(answer_num - correct_num) / correct_num < 0.05
            except:
                # For text answers, check if normalized answer is contained in correct answer
                correct = (answer in correct_norm) or (correct_norm in answer)

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

    def _load_cached_questions(self) -> List[GPQAQuestion]:
        """Load questions from cached pickle file."""
        import pickle

        cache_file = self.dataset_path / "gpqa_questions.pkl"
        if not cache_file.exists():
            raise FileNotFoundError("Cached questions not found")

        with open(cache_file, "rb") as f:
            return pickle.load(f)

    def _cache_questions(self) -> None:
        """Cache questions to pickle file."""
        import pickle

        self.dataset_path.mkdir(parents=True, exist_ok=True)
        cache_file = self.dataset_path / "gpqa_questions.pkl"

        with open(cache_file, "wb") as f:
            pickle.dump(self.questions, f)


class LLMBenchmarkSuite:
    def __init__(
        self,
        model_name: str = "llama3.2:latest",
        temperature: float = 0.5,
        top_p: float = 1.0,
        top_k: int = 1,
        questions_per_task: int = 100,
        max_tokens: int = 50,
        context_window: int = 512,
        timeout: float = 10.0,
    ):
        self.model_name = model_name
        self.questions_per_task = questions_per_task
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.timeout = timeout
        self.tasks: Dict[BenchmarkType, List[BenchmarkTask]] = {}
        self.results: Dict[BenchmarkType, List[BenchmarkResult]] = {}

        self.client = AsyncClient()
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.logger = logging.getLogger(__name__)

    async def add_benchmark(
        self, benchmark_type: BenchmarkType, tasks: List[BenchmarkTask]
    ) -> None:
        """Add a benchmark type and its associated tasks."""
        if not tasks:
            raise ValueError(f"No tasks provided for benchmark type {benchmark_type}")

        self.tasks[benchmark_type] = tasks
        self.logger.info(
            f"Added {len(tasks)} tasks for benchmark type {benchmark_type.name}"
        )

    async def _get_llm_response(self, prompt: str, system_prompt: str = "") -> str:
        """Get single response from Ollama LLM with timeout."""
        try:
            async with asyncio.timeout(self.timeout):
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                response = await self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    options={
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": self.top_k,
                        "num_predict": self.max_tokens,
                        "context_length": self.context_window,
                        "stop": ["\n", "Question:", "Options:"],
                    },
                )
                return response["message"]["content"].strip()
        except asyncio.TimeoutError:
            self.logger.warning(f"Request timed out after {self.timeout}s")
            return ""
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            return ""

    async def run_benchmarks(
        self, selected_types: Optional[List[BenchmarkType]] = None
    ) -> Dict[str, float]:
        """Run selected benchmarks sequentially."""
        types_to_run = selected_types or list(self.tasks.keys())
        scores = {}

        self.logger.info(
            f"Starting benchmarks for types: {[t.name for t in types_to_run]}"
        )

        for benchmark_type in tqdm(types_to_run, desc="Running benchmarks"):
            if benchmark_type not in self.tasks:
                self.logger.warning(
                    f"No tasks found for benchmark type {benchmark_type.name}"
                )
                continue

            tasks = self.tasks[benchmark_type]
            self.logger.info(f"Running {len(tasks)} tasks for {benchmark_type.name}")
            results = []

            for task in tqdm(
                tasks, desc=f"Running {benchmark_type.name} tasks", leave=False
            ):
                task_results = []
                for _ in tqdm(
                    range(self.questions_per_task),
                    desc=f"Processing questions",
                    leave=False,
                ):
                    prompt = await task.generate_prompt()
                    system_prompt = getattr(task, "system_prompt", "")
                    response = await self._get_llm_response(prompt, system_prompt)
                    result = await task.evaluate(response)
                    task_results.append(result)

                self.logger.info(
                    f"Task completed with avg score: {sum(r.score for r in task_results) / len(task_results):.2f}"
                )
                results.extend(task_results)

            self.results[benchmark_type] = results
            scores[benchmark_type.name] = sum(r.score for r in results) / len(results)
            self.logger.info(
                f"Benchmark {benchmark_type.name} completed with score: {scores[benchmark_type.name]:.2f}"
            )

        return scores

    async def generate_report(self) -> str:
        """Generate a detailed report of benchmark results."""
        if not self.results:
            return "No benchmark results available."

        report = []
        summary_stats = {}
        
        for benchmark_type, results in self.results.items():
            if not results:
                continue

            total_score = sum(r.score for r in results)
            avg_score = total_score / len(results)
            
            # Calculate additional statistics
            scores = [r.score for r in results]
            summary_stats[benchmark_type.name] = {
                "average": avg_score,
                "total_questions": len(results),
                "correct_answers": sum(1 for r in results if r.score > 0),
                "subjects": set(r.metadata.get("subject", "Unknown") for r in results),
                "strengths": set(s for r in results for s in r.strengths),
                "weaknesses": set(w for r in results for w in r.weaknesses)
            }

            report.append(f"\n{benchmark_type.name} Benchmark Results:")
            report.append(f"Average Score: {avg_score:.2f}")
            report.append(f"Total Questions: {len(results)}")

            # Add some example questions and responses
            report.append("\nSample Questions and Responses:")
            for i, result in enumerate(results[:3], 1):
                report.append(f"\nExample {i}:")
                report.append(f"Question: {result.metadata['question']}")
                report.append(f"Model Answer: {result.metadata['model_answer']}")
                report.append(f"Correct Answer: {result.metadata['correct_answer']}")
                report.append(f"Score: {result.score}")

        # Add summary section
        report.append("\n" + "="*50)
        report.append("BENCHMARK SUMMARY")
        report.append("="*50)
        
        for benchmark, stats in summary_stats.items():
            report.append(f"\n{benchmark}:")
            report.append(f"- Score: {stats['average']:.2f}")
            report.append(f"- Questions: {stats['total_questions']}")
            report.append(f"- Correct Answers: {stats['correct_answers']} ({(stats['correct_answers']/stats['total_questions']*100):.1f}%)")
            report.append(f"- Subjects Covered: {len(stats['subjects'])}")
            
            if stats['strengths']:
                report.append("- Key Strengths:")
                for strength in list(stats['strengths'])[:3]:  # Top 3 strengths
                    report.append(f"  * {strength}")
                    
            if stats['weaknesses']:
                report.append("- Common Errors:")
                for weakness in list(stats['weaknesses'])[:3]:  # Top 3 weaknesses
                    report.append(f"  * {weakness}")
                    
        report.append("\nOverall Performance:")
        overall_score = sum(s['average'] for s in summary_stats.values()) / len(summary_stats)
        report.append(f"Average Score Across All Benchmarks: {overall_score:.2f}")
        
        return "\n".join(report)


# Example usage:
async def main():
    suite = LLMBenchmarkSuite(
        model_name="qwen2.5-coder:32b",
        questions_per_task=100,
        max_tokens=100,
        context_window=512,
        temperature=0.0,
        timeout=10.0,
    )
    logger = logging.getLogger(__name__)

    with tqdm(total=5, desc="Benchmark Setup") as pbar:
        try:
            # Initialize MMLU
            mmlu_task = MMluTask(download=True)
            await mmlu_task.initialize()
            pbar.update(1)

            # Initialize GPQA tasks separately
            gpqa_tasks = []
            for config in ["gpqa_main", "gpqa_extended"]:
                task = GPQATask(
                    domains=[config]
                )  # Create separate task for each config
                await task.initialize()
                gpqa_tasks.append(task)
            pbar.update(1)

            # Add benchmarks
            await suite.add_benchmark(BenchmarkType.MMLU, [mmlu_task])
            await suite.add_benchmark(
                BenchmarkType.GPQA, gpqa_tasks  # Add both GPQA tasks
            )
            pbar.update(1)

            # Run benchmarks
            scores = await suite.run_benchmarks(
                [BenchmarkType.MMLU, BenchmarkType.GPQA]
            )
            pbar.update(1)

            # Generate report
            report = await suite.generate_report()
            pbar.update(1)

            print("\nBenchmark Results:")
            print(json.dumps(scores, indent=2))
            print("\nDetailed Report:")
            print(report)

        except Exception as e:
            logger.error(f"Error during benchmark setup/execution: {e}")
            raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Silence httpx logging
    logging.getLogger("httpx").setLevel(logging.WARNING)

    asyncio.run(main())
