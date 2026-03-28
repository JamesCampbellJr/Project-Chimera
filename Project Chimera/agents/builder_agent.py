# /agents/builder_agent.py

"""
Builder Agent — Constructs software projects on demand.

Uses the CodeBuilder and ProjectManager to plan, generate, and deliver
complete software projects in multiple languages and frameworks.
"""

import asyncio
import logging
import os
from typing import Optional

from agent import Agent
from core.code_builder import CodeBuilder
from core.project_manager import ProjectManager
from core.self_improvement import SelfImprovement
import config


class BuilderAgent(Agent):
    """
    Specialist agent that builds software projects.

    Given a project description, it plans the project, generates all files,
    reviews them for quality, and improves them based on feedback.
    """

    def __init__(
        self,
        orchestrator,
        project_description: str,
        memory_store=None,
    ):
        super().__init__(
            orchestrator,
            role=config.BUILDER_ROLE,
            agent_id=f"Builder-{os.getpid()}",
        )
        self.project_description = project_description
        self.memory = memory_store
        self.code_builder = CodeBuilder(memory_store=memory_store)
        self.project_manager = ProjectManager(memory_store=memory_store)
        self.self_improvement = SelfImprovement(memory_store=memory_store)

    async def process_task(self, task: str):
        """
        Full project build pipeline:
        1. Plan the project
        2. Build all files
        3. Review the output
        4. Improve based on review
        5. Report results
        """
        description = task or self.project_description
        logging.info("BuilderAgent starting project: %s", description)

        try:
            # 1. Create project and plan
            project = await self.project_manager.create_project(description)
            logging.info("Project plan created: %s", project.id)

            # 2. Build the project
            project.update_status("in_progress")
            result = await self.code_builder.build_project(
                description, project.output_dir
            )

            if result["status"] != "success":
                project.update_status("failed")
                self.project_manager.add_error(project.id, "Build failed")
                logging.error("Project build failed for %s", project.id)
                await self.stop()
                return

            project.files_created = result["files_created"]

            # 3. Self-review cycle — evaluate each generated file
            for file_path in result["files_created"]:
                full_path = os.path.join(project.output_dir, file_path)
                if not os.path.exists(full_path):
                    continue

                with open(full_path, "r") as fh:
                    code = fh.read()

                evaluation = await self.self_improvement.evaluate_output(
                    task_description=f"Generate {file_path} for: {description}",
                    output=code,
                    output_type="code",
                )
                score = evaluation.get("overall_score", 0.5)

                # 4. Improve files scoring below threshold
                if isinstance(score, (int, float)) and score < 0.7:
                    feedback = "; ".join(evaluation.get("suggestions", []))
                    if feedback:
                        logging.info("Improving %s (score: %.2f)", file_path, score)
                        improved = await self.code_builder.improve_code(
                            code, file_path, feedback
                        )
                        with open(full_path, "w") as fh:
                            fh.write(improved)

                self.project_manager.complete_task(
                    project.id, f"Generate {file_path}"
                )

            # 5. Mark complete
            project.update_status("completed")
            self.project_manager._save_projects()

            # Store success in memory
            if self.memory:
                self.memory.store(
                    content=(
                        f"Successfully built project: {description}. "
                        f"Files: {', '.join(result['files_created'])}. "
                        f"Output: {project.output_dir}"
                    ),
                    category="skills",
                    tags=["project_build", "completed"],
                    metadata={"project_id": project.id, "files": result["files_created"]},
                    importance=0.8,
                )

            logging.info(
                "BuilderAgent completed project %s with %d files.",
                project.id,
                len(result["files_created"]),
            )

            # Report back to orchestrator
            await self.orchestrator.report_skill_learned(
                f"build_{description.replace(' ', '_')[:30]}",
                project.output_dir,
            )

        except Exception as exc:
            logging.error("BuilderAgent failed: %s", exc, exc_info=True)
            if self.memory:
                self.memory.store(
                    content=f"Build failed for: {description}. Error: {exc}",
                    category="errors",
                    tags=["build_failure"],
                    metadata={"description": description, "error": str(exc)},
                    importance=0.9,
                )

        await self.stop()
