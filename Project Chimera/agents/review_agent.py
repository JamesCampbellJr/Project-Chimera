# /agents/review_agent.py

"""
Code Review Agent — Provides quality assurance for generated code.

Reviews code outputs from other agents, identifies issues, and triggers
improvements through the self-improvement feedback loop.
"""

import asyncio
import logging
import os
from typing import Optional

from agent import Agent
from core.self_improvement import SelfImprovement
from core.code_builder import CodeBuilder
import config


class ReviewAgent(Agent):
    """
    Specialist agent that reviews code quality and triggers improvements.
    """

    def __init__(self, orchestrator, memory_store=None):
        super().__init__(
            orchestrator,
            role=config.REVIEW_ROLE,
            agent_id=f"Reviewer-{os.getpid()}",
        )
        self.memory = memory_store
        self.self_improvement = SelfImprovement(memory_store=memory_store)
        self.code_builder = CodeBuilder(memory_store=memory_store)

    async def process_task(self, task: str):
        """
        Review a project directory or specific file.

        task should be a path to a file or directory to review.
        """
        logging.info("ReviewAgent reviewing: %s", task)

        try:
            if os.path.isfile(task):
                await self._review_file(task)
            elif os.path.isdir(task):
                await self._review_directory(task)
            else:
                logging.warning("ReviewAgent: '%s' is not a valid file or directory.", task)
        except Exception as exc:
            logging.error("ReviewAgent error: %s", exc, exc_info=True)

        # Generate improvement plan based on all reviews
        plan = await self.self_improvement.generate_improvement_plan()
        logging.info("Improvement plan: %s", plan.get("summary", "N/A"))

        await self.stop()

    async def _review_file(self, file_path: str):
        """Review a single file."""
        try:
            with open(file_path, "r") as fh:
                code = fh.read()
        except (OSError, UnicodeDecodeError) as exc:
            logging.error("Cannot read file %s: %s", file_path, exc)
            return

        evaluation = await self.self_improvement.evaluate_output(
            task_description=f"Review {os.path.basename(file_path)}",
            output=code,
            output_type="code",
        )

        score = evaluation.get("overall_score", 0.5)
        logging.info("File %s scored %.2f", file_path, score if isinstance(score, (int, float)) else 0.5)

        # Improve if below threshold
        if isinstance(score, (int, float)) and score < 0.7:
            suggestions = evaluation.get("suggestions", [])
            if suggestions:
                feedback = "; ".join(suggestions)
                logging.info("Improving %s based on review feedback.", file_path)
                improved = await self.code_builder.improve_code(code, file_path, feedback)
                with open(file_path, "w") as fh:
                    fh.write(improved)

    async def _review_directory(self, dir_path: str):
        """Review all code files in a directory."""
        code_extensions = {".py", ".rs", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".go", ".java"}

        for root, _dirs, files in os.walk(dir_path):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in code_extensions:
                    file_path = os.path.join(root, filename)
                    await self._review_file(file_path)
