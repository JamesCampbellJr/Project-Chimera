# /core/code_builder.py

"""
Code Generation and Project Building Engine.

Uses LLM-based code generation to create software projects in multiple
languages and frameworks (Python, Rust, JavaScript/TypeScript, HTML/CSS, etc.).
Works with the memory system to learn from past projects and improve over time.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import ollama

import config

# Mapping of project types to their scaffolding templates
PROJECT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "python_cli": {
        "description": "A Python command-line application",
        "files": ["main.py", "requirements.txt", "README.md", ".gitignore"],
        "language": "python",
    },
    "python_web": {
        "description": "A Python web application using Flask or FastAPI",
        "files": [
            "app.py",
            "requirements.txt",
            "README.md",
            "templates/index.html",
            "static/style.css",
            ".gitignore",
        ],
        "language": "python",
    },
    "rust_cli": {
        "description": "A Rust command-line application",
        "files": ["src/main.rs", "Cargo.toml", "README.md", ".gitignore"],
        "language": "rust",
    },
    "website": {
        "description": "A static website with HTML, CSS, and JavaScript",
        "files": ["index.html", "styles.css", "script.js", "README.md"],
        "language": "html",
    },
    "react_app": {
        "description": "A React web application",
        "files": [
            "package.json",
            "public/index.html",
            "src/index.js",
            "src/App.js",
            "src/App.css",
            "README.md",
            ".gitignore",
        ],
        "language": "javascript",
    },
    "node_api": {
        "description": "A Node.js REST API",
        "files": [
            "package.json",
            "src/index.js",
            "src/routes.js",
            "README.md",
            ".gitignore",
        ],
        "language": "javascript",
    },
}

SUPPORTED_LANGUAGES = ("python", "rust", "javascript", "typescript", "html", "css", "go", "java")


class CodeBuilder:
    """
    Generates code and builds software projects using LLM-based generation.
    """

    def __init__(self, memory_store=None):
        self.memory = memory_store
        logging.info("CodeBuilder initialized.")

    async def classify_project(self, description: str) -> Dict[str, Any]:
        """
        Analyze a project description and determine the best project type,
        language, and architecture.
        """
        prompt = f"""You are an expert software architect. Analyze this project request and decide the best approach.

Project request: "{description}"

Respond with a JSON object containing:
- "project_type": one of {list(PROJECT_TEMPLATES.keys())} or "custom"
- "language": primary language ({', '.join(SUPPORTED_LANGUAGES)})
- "framework": recommended framework (e.g., "flask", "fastapi", "react", "actix-web", or "none")
- "files": list of file paths to create
- "architecture": brief architecture description
- "complexity": "simple", "medium", or "complex"

Respond ONLY with the JSON object."""

        try:
            response = await ollama.AsyncClient().chat(
                model=config.COGNITIVE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
            result = json.loads(response["message"]["content"])
            logging.info("Project classified: %s", result.get("project_type"))
            return result
        except (json.JSONDecodeError, KeyError) as exc:
            logging.error("Failed to classify project: %s", exc)
            return {
                "project_type": "custom",
                "language": "python",
                "framework": "none",
                "files": ["main.py", "README.md"],
                "architecture": "Simple single-file application",
                "complexity": "simple",
            }

    async def generate_file_content(
        self,
        file_path: str,
        project_description: str,
        project_plan: Dict[str, Any],
        existing_files: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Generate the content for a single file in the project.
        """
        context_parts = [
            f'Project description: "{project_description}"',
            f"Architecture: {project_plan.get('architecture', 'N/A')}",
            f"Language: {project_plan.get('language', 'N/A')}",
            f"Framework: {project_plan.get('framework', 'none')}",
        ]

        if existing_files:
            context_parts.append("Already created files and their contents:")
            for fp, content in existing_files.items():
                context_parts.append(f"\n--- {fp} ---\n{content[:2000]}")

        # Pull relevant memories for context
        if self.memory:
            memories = self.memory.search(
                project_description,
                category="skills",
                limit=3,
            )
            if memories:
                context_parts.append("\nRelevant past experience:")
                for mem in memories:
                    context_parts.append(f"- {mem.content}")

        prompt = f"""{chr(10).join(context_parts)}

Generate the complete content for the file: "{file_path}"

Requirements:
- Write production-quality code with proper error handling
- Include appropriate comments and documentation
- Follow best practices for the language/framework
- Make the code functional and ready to run

Respond with ONLY the file content, no explanations or markdown code fences."""

        try:
            response = await ollama.AsyncClient().chat(
                model=config.COGNITIVE_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response["message"]["content"].strip()
            # Strip markdown fences if the model includes them
            if content.startswith("```"):
                lines = content.split("\n")
                # Remove first line (```lang) and last line (```)
                if lines[-1].strip() == "```":
                    lines = lines[1:-1]
                else:
                    lines = lines[1:]
                content = "\n".join(lines)
            return content
        except Exception as exc:
            logging.error("Failed to generate content for %s: %s", file_path, exc)
            return f"# Error generating content for {file_path}\n# {exc}\n"

    async def build_project(
        self,
        description: str,
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        Build a complete software project from a description.

        Returns a dict with project metadata and generated files.
        """
        logging.info("Building project: %s", description)

        # 1. Classify the project
        plan = await self.classify_project(description)
        logging.info("Project plan: %s", json.dumps(plan, indent=2))

        # 2. Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # 3. Generate files
        generated_files: Dict[str, str] = {}
        files_to_create = plan.get("files", [])

        for file_path in files_to_create:
            logging.info("Generating: %s", file_path)
            content = await self.generate_file_content(
                file_path, description, plan, generated_files
            )
            generated_files[file_path] = content

            # Write to disk
            full_path = os.path.join(output_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as fh:
                fh.write(content)
            logging.info("Written: %s", full_path)

        # 4. Store in memory
        if self.memory:
            self.memory.store(
                content=f"Built project: {description}. Type: {plan.get('project_type')}. "
                f"Language: {plan.get('language')}. Files: {', '.join(files_to_create)}",
                category="projects",
                tags=[plan.get("language", ""), plan.get("project_type", ""), "completed"],
                metadata={"output_dir": output_dir, "plan": plan},
                importance=0.7,
            )

        result = {
            "status": "success",
            "output_dir": output_dir,
            "plan": plan,
            "files_created": list(generated_files.keys()),
            "file_count": len(generated_files),
        }
        logging.info("Project build complete: %d files created", len(generated_files))
        return result

    async def improve_code(
        self,
        code: str,
        file_path: str,
        feedback: str,
    ) -> str:
        """
        Improve existing code based on feedback.
        """
        prompt = f"""You are an expert code reviewer and improver.

File: {file_path}
Feedback: {feedback}

Current code:
```
{code}
```

Improve the code based on the feedback. Maintain the same functionality but make it better.
Respond with ONLY the improved code, no explanations or markdown code fences."""

        try:
            response = await ollama.AsyncClient().chat(
                model=config.COGNITIVE_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            improved = response["message"]["content"].strip()
            if improved.startswith("```"):
                lines = improved.split("\n")
                if lines[-1].strip() == "```":
                    lines = lines[1:-1]
                else:
                    lines = lines[1:]
                improved = "\n".join(lines)
            return improved
        except Exception as exc:
            logging.error("Failed to improve code for %s: %s", file_path, exc)
            return code
