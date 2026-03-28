# /core/self_improvement.py

"""
Self-Improvement Module for Project Chimera.

Evaluates agent outputs, learns from successes and failures, and generates
improvement suggestions. Works with the memory system to build a persistent
feedback loop that makes the system better over time.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import ollama

import config


class SelfImprovement:
    """
    Evaluates agent performance, captures lessons learned,
    and generates improvement plans.
    """

    def __init__(self, memory_store=None):
        self.memory = memory_store
        self.evaluation_history: List[Dict[str, Any]] = []
        logging.info("SelfImprovement module initialized.")

    async def evaluate_output(
        self,
        task_description: str,
        output: str,
        output_type: str = "code",
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of an agent's output and provide scored feedback.

        Args:
            task_description: What the agent was asked to do.
            output: The actual output produced.
            output_type: Type of output ("code", "plan", "text", "project").

        Returns:
            A dict with quality scores, issues found, and suggestions.
        """
        prompt = f"""You are an expert quality evaluator. Evaluate this {output_type} output.

Task: "{task_description}"

Output to evaluate:
```
{output[:4000]}
```

Rate the output on these criteria (0.0 to 1.0):
- "correctness": Does it correctly address the task?
- "completeness": Does it fully solve the problem?
- "code_quality": Is it well-structured and following best practices?
- "error_handling": Does it handle edge cases and errors?
- "documentation": Is it well-documented?

Also provide:
- "issues": List of specific problems found
- "suggestions": List of concrete improvements
- "overall_score": Weighted average score (0.0 to 1.0)
- "summary": One-sentence summary

Respond ONLY with a JSON object."""

        try:
            response = await ollama.AsyncClient().chat(
                model=config.COGNITIVE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
            evaluation = json.loads(response["message"]["content"])
        except (json.JSONDecodeError, Exception) as exc:
            logging.error("Failed to evaluate output: %s", exc)
            evaluation = {
                "correctness": 0.5,
                "completeness": 0.5,
                "code_quality": 0.5,
                "error_handling": 0.5,
                "documentation": 0.5,
                "issues": [f"Evaluation failed: {exc}"],
                "suggestions": [],
                "overall_score": 0.5,
                "summary": "Evaluation could not be completed.",
            }

        # Record evaluation
        record = {
            "task": task_description,
            "output_type": output_type,
            "evaluation": evaluation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.evaluation_history.append(record)

        # Store feedback in memory for future learning
        if self.memory:
            score = evaluation.get("overall_score", 0.5)
            issues = evaluation.get("issues", [])
            suggestions = evaluation.get("suggestions", [])

            self.memory.store(
                content=(
                    f"Task: {task_description}. Score: {score}. "
                    f"Issues: {'; '.join(issues[:3]) if issues else 'None'}. "
                    f"Suggestions: {'; '.join(suggestions[:3]) if suggestions else 'None'}"
                ),
                category="feedback",
                tags=[output_type, "evaluation"],
                metadata={"score": score, "evaluation": evaluation},
                importance=0.8 if score < 0.6 else 0.5,
            )

        return evaluation

    async def generate_improvement_plan(self) -> Dict[str, Any]:
        """
        Analyze recent evaluations and generate a plan for system improvement.
        """
        if not self.evaluation_history and not self.memory:
            return {"improvements": [], "priority": "none", "summary": "No data to analyze."}

        # Gather recent feedback
        recent_feedback = []
        if self.memory:
            feedback_entries = self.memory.get_by_category("feedback", limit=20)
            recent_feedback = [e.content for e in feedback_entries]

        # Also use in-session history
        for record in self.evaluation_history[-10:]:
            ev = record.get("evaluation", {})
            recent_feedback.append(
                f"Task: {record['task']}. Score: {ev.get('overall_score', 'N/A')}"
            )

        if not recent_feedback:
            return {"improvements": [], "priority": "none", "summary": "No feedback data available."}

        prompt = f"""You are an AI system improvement analyst. Based on this feedback data,
identify patterns and generate an improvement plan.

Recent feedback:
{json.dumps(recent_feedback[:15], indent=2)}

Generate a JSON object with:
- "patterns": List of recurring issues or patterns
- "improvements": List of specific, actionable improvements (each with "action", "priority", "impact")
- "strengths": What the system does well
- "summary": Brief overall assessment

Respond ONLY with a JSON object."""

        try:
            response = await ollama.AsyncClient().chat(
                model=config.COGNITIVE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
            plan = json.loads(response["message"]["content"])
        except (json.JSONDecodeError, Exception) as exc:
            logging.error("Failed to generate improvement plan: %s", exc)
            plan = {
                "patterns": [],
                "improvements": [],
                "strengths": [],
                "summary": f"Analysis failed: {exc}",
            }

        # Store the improvement plan in memory
        if self.memory:
            self.memory.store(
                content=f"Improvement plan generated: {plan.get('summary', 'N/A')}",
                category="improvements",
                tags=["improvement_plan", "self_analysis"],
                metadata=plan,
                importance=0.9,
            )

        return plan

    def get_performance_summary(self) -> Dict[str, Any]:
        """Return a summary of recent performance metrics."""
        if not self.evaluation_history:
            return {"total_evaluations": 0, "average_score": 0.0, "message": "No evaluations yet."}

        scores = []
        for record in self.evaluation_history:
            score = record.get("evaluation", {}).get("overall_score")
            if isinstance(score, (int, float)):
                scores.append(score)

        avg = sum(scores) / len(scores) if scores else 0.0
        return {
            "total_evaluations": len(self.evaluation_history),
            "average_score": round(avg, 3),
            "recent_scores": scores[-10:],
            "trend": "improving" if len(scores) >= 3 and scores[-1] > scores[0] else "stable",
        }
