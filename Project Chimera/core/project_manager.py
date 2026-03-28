# /core/project_manager.py

"""
Project Management Module for Project Chimera.

Coordinates multi-step project planning and execution, tracks project state,
and manages the lifecycle of software projects from inception to delivery.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import ollama

import config


class ProjectState:
    """Tracks the state of a project through its lifecycle."""

    STATUSES = ("planning", "in_progress", "review", "completed", "failed")

    def __init__(self, project_id: str, description: str):
        self.id = project_id
        self.description = description
        self.status = "planning"
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at
        self.plan: Dict[str, Any] = {}
        self.files_created: List[str] = []
        self.tasks_completed: List[str] = []
        self.tasks_pending: List[str] = []
        self.errors: List[str] = []
        self.output_dir = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "plan": self.plan,
            "files_created": self.files_created,
            "tasks_completed": self.tasks_completed,
            "tasks_pending": self.tasks_pending,
            "errors": self.errors,
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectState":
        proj = cls(data["id"], data["description"])
        proj.status = data.get("status", "planning")
        proj.created_at = data.get("created_at", proj.created_at)
        proj.updated_at = data.get("updated_at", proj.updated_at)
        proj.plan = data.get("plan", {})
        proj.files_created = data.get("files_created", [])
        proj.tasks_completed = data.get("tasks_completed", [])
        proj.tasks_pending = data.get("tasks_pending", [])
        proj.errors = data.get("errors", [])
        proj.output_dir = data.get("output_dir", "")
        return proj

    def update_status(self, status: str):
        if status not in self.STATUSES:
            raise ValueError(f"Invalid status '{status}'. Must be one of {self.STATUSES}")
        self.status = status
        self.updated_at = datetime.now(timezone.utc).isoformat()


class ProjectManager:
    """
    Manages the full lifecycle of software projects.

    Handles project planning, task decomposition, progress tracking,
    and project persistence.
    """

    PROJECTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "projects")

    def __init__(self, memory_store=None, projects_dir: Optional[str] = None):
        self.memory = memory_store
        self.projects_dir = projects_dir or self.PROJECTS_DIR
        self.active_projects: Dict[str, ProjectState] = {}
        os.makedirs(self.projects_dir, exist_ok=True)
        self._load_projects()
        logging.info("ProjectManager initialized with %d projects.", len(self.active_projects))

    def _load_projects(self):
        """Load project states from disk."""
        state_file = os.path.join(self.projects_dir, "_project_states.json")
        if not os.path.exists(state_file):
            return
        try:
            with open(state_file, "r") as fh:
                data = json.load(fh)
            for proj_data in data:
                proj = ProjectState.from_dict(proj_data)
                self.active_projects[proj.id] = proj
        except (json.JSONDecodeError, KeyError) as exc:
            logging.error("Failed to load project states: %s", exc)

    def _save_projects(self):
        """Persist project states to disk."""
        state_file = os.path.join(self.projects_dir, "_project_states.json")
        data = [proj.to_dict() for proj in self.active_projects.values()]
        with open(state_file, "w") as fh:
            json.dump(data, fh, indent=2)

    async def create_project(self, description: str) -> ProjectState:
        """
        Create a new project and generate a development plan.
        """
        project_id = f"proj_{int(datetime.now(timezone.utc).timestamp())}"
        project = ProjectState(project_id, description)
        project.output_dir = os.path.join(self.projects_dir, project_id)
        os.makedirs(project.output_dir, exist_ok=True)

        # Generate a detailed plan
        plan = await self._generate_plan(description)
        project.plan = plan
        project.tasks_pending = plan.get("tasks", [])

        self.active_projects[project_id] = project
        self._save_projects()

        # Store in memory
        if self.memory:
            self.memory.store(
                content=f"New project created: {description}. ID: {project_id}",
                category="projects",
                tags=["new_project", plan.get("language", "unknown")],
                metadata={"project_id": project_id, "plan_summary": plan.get("summary", "")},
                importance=0.8,
            )

        logging.info("Project created: %s (%s)", project_id, description)
        return project

    async def _generate_plan(self, description: str) -> Dict[str, Any]:
        """Generate a detailed development plan using LLM."""
        # Fetch relevant past project memories
        context = ""
        if self.memory:
            past_projects = self.memory.search(description, category="projects", limit=3)
            if past_projects:
                context = "Relevant past projects:\n" + "\n".join(
                    f"- {m.content}" for m in past_projects
                )

        prompt = f"""You are a senior project manager. Create a development plan for this project.

Project: "{description}"
{context}

Generate a JSON object with:
- "summary": Brief project summary
- "language": Primary programming language
- "framework": Recommended framework
- "tasks": List of ordered task descriptions (strings) to complete the project
- "files": List of file paths to create
- "estimated_complexity": "simple", "medium", or "complex"
- "dependencies": List of external dependencies/packages needed

Respond ONLY with a JSON object."""

        try:
            response = await ollama.AsyncClient().chat(
                model=config.COGNITIVE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
            return json.loads(response["message"]["content"])
        except (json.JSONDecodeError, Exception) as exc:
            logging.error("Failed to generate project plan: %s", exc)
            return {
                "summary": description,
                "language": "python",
                "framework": "none",
                "tasks": [f"Build: {description}"],
                "files": ["main.py", "README.md"],
                "estimated_complexity": "medium",
                "dependencies": [],
            }

    def complete_task(self, project_id: str, task: str):
        """Mark a task as completed in a project."""
        project = self.active_projects.get(project_id)
        if not project:
            logging.warning("Project %s not found.", project_id)
            return
        if task in project.tasks_pending:
            project.tasks_pending.remove(task)
        project.tasks_completed.append(task)
        project.update_status(
            "completed" if not project.tasks_pending else "in_progress"
        )
        self._save_projects()

    def add_error(self, project_id: str, error: str):
        """Record an error that occurred during project execution."""
        project = self.active_projects.get(project_id)
        if project:
            project.errors.append(error)
            self._save_projects()

    def get_project(self, project_id: str) -> Optional[ProjectState]:
        """Retrieve a project by ID."""
        return self.active_projects.get(project_id)

    def list_projects(self, status: Optional[str] = None) -> List[ProjectState]:
        """List all projects, optionally filtered by status."""
        projects = list(self.active_projects.values())
        if status:
            projects = [p for p in projects if p.status == status]
        return sorted(projects, key=lambda p: p.created_at, reverse=True)
