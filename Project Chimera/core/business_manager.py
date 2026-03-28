# /core/business_manager.py

"""
Business Management Module for Project Chimera.

Tracks revenue, manages client projects, generates proposals,
and identifies business opportunities. Designed to help the system
work towards profitability by managing the business side of operations.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import ollama

import config


class ClientProject:
    """Represents a client project with pricing and status tracking."""

    def __init__(
        self,
        client_name: str,
        description: str,
        price: float,
        project_id: Optional[str] = None,
    ):
        self.id = project_id or f"client_{int(datetime.now(timezone.utc).timestamp())}"
        self.client_name = client_name
        self.description = description
        self.price = price
        self.status = "proposal"  # proposal, accepted, in_progress, delivered, paid
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.delivered_at: Optional[str] = None
        self.paid_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "client_name": self.client_name,
            "description": self.description,
            "price": self.price,
            "status": self.status,
            "created_at": self.created_at,
            "delivered_at": self.delivered_at,
            "paid_at": self.paid_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClientProject":
        proj = cls(
            client_name=data["client_name"],
            description=data["description"],
            price=data["price"],
            project_id=data.get("id"),
        )
        proj.status = data.get("status", "proposal")
        proj.created_at = data.get("created_at", proj.created_at)
        proj.delivered_at = data.get("delivered_at")
        proj.paid_at = data.get("paid_at")
        return proj


class BusinessManager:
    """
    Manages the business operations of the AI company.

    Tracks clients, revenue, generates proposals, and identifies
    opportunities for growth.
    """

    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "business_data")

    def __init__(self, memory_store=None, data_dir: Optional[str] = None):
        self.memory = memory_store
        self.data_dir = data_dir or self.DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)
        self.client_projects: Dict[str, ClientProject] = {}
        self.revenue_log: List[Dict[str, Any]] = []
        self._load_data()
        logging.info("BusinessManager initialized.")

    def _data_path(self, filename: str) -> str:
        return os.path.join(self.data_dir, filename)

    def _load_data(self):
        """Load business data from disk."""
        # Load client projects
        projects_path = self._data_path("client_projects.json")
        if os.path.exists(projects_path):
            try:
                with open(projects_path, "r") as fh:
                    data = json.load(fh)
                for item in data:
                    proj = ClientProject.from_dict(item)
                    self.client_projects[proj.id] = proj
            except (json.JSONDecodeError, KeyError) as exc:
                logging.error("Failed to load client projects: %s", exc)

        # Load revenue log
        revenue_path = self._data_path("revenue_log.json")
        if os.path.exists(revenue_path):
            try:
                with open(revenue_path, "r") as fh:
                    self.revenue_log = json.load(fh)
            except json.JSONDecodeError as exc:
                logging.error("Failed to load revenue log: %s", exc)

    def _save_data(self):
        """Persist business data to disk."""
        # Save client projects
        projects_path = self._data_path("client_projects.json")
        with open(projects_path, "w") as fh:
            json.dump(
                [p.to_dict() for p in self.client_projects.values()],
                fh,
                indent=2,
            )

        # Save revenue log
        revenue_path = self._data_path("revenue_log.json")
        with open(revenue_path, "w") as fh:
            json.dump(self.revenue_log, fh, indent=2)

    async def generate_proposal(
        self,
        client_name: str,
        project_description: str,
    ) -> Dict[str, Any]:
        """
        Generate a professional project proposal with pricing.
        """
        # Use past project data for better estimates
        context = ""
        if self.memory:
            past = self.memory.search(project_description, category="projects", limit=3)
            if past:
                context = "Similar past projects:\n" + "\n".join(
                    f"- {m.content}" for m in past
                )

        prompt = f"""You are a professional software development company creating a proposal.

Client: {client_name}
Project request: "{project_description}"
{context}

Generate a JSON proposal with:
- "title": Professional project title
- "summary": Executive summary (2-3 sentences)
- "scope": List of deliverables
- "timeline_days": Estimated days to complete
- "price_usd": Recommended price in USD (be competitive but fair)
- "tech_stack": Recommended technology stack
- "phases": List of project phases with descriptions

Respond ONLY with a JSON object."""

        try:
            response = await ollama.AsyncClient().chat(
                model=config.COGNITIVE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
            proposal = json.loads(response["message"]["content"])
        except (json.JSONDecodeError, Exception) as exc:
            logging.error("Failed to generate proposal: %s", exc)
            proposal = {
                "title": f"Project for {client_name}",
                "summary": project_description,
                "scope": [project_description],
                "timeline_days": 14,
                "price_usd": 2500,
                "tech_stack": ["Python"],
                "phases": [{"phase": "Development", "description": project_description}],
            }

        # Create the client project record
        price = proposal.get("price_usd", 2500)
        client_proj = ClientProject(client_name, project_description, price)
        self.client_projects[client_proj.id] = client_proj
        self._save_data()

        # Store in memory
        if self.memory:
            self.memory.store(
                content=(
                    f"Proposal for {client_name}: {project_description}. "
                    f"Price: ${price}. Timeline: {proposal.get('timeline_days', 'N/A')} days."
                ),
                category="business",
                tags=["proposal", client_name.lower().replace(" ", "_")],
                metadata={"proposal": proposal, "client_project_id": client_proj.id},
                importance=0.8,
            )

        proposal["client_project_id"] = client_proj.id
        return proposal

    def record_payment(self, project_id: str, amount: float):
        """Record a payment received for a client project."""
        proj = self.client_projects.get(project_id)
        if not proj:
            logging.warning("Client project %s not found.", project_id)
            return

        proj.status = "paid"
        proj.paid_at = datetime.now(timezone.utc).isoformat()

        self.revenue_log.append(
            {
                "project_id": project_id,
                "client": proj.client_name,
                "amount": amount,
                "date": proj.paid_at,
            }
        )
        self._save_data()

        if self.memory:
            self.memory.store(
                content=f"Payment received: ${amount} from {proj.client_name} for: {proj.description}",
                category="business",
                tags=["payment", "revenue"],
                metadata={"amount": amount, "project_id": project_id},
                importance=0.9,
            )

        logging.info("Payment of $%.2f recorded for project %s.", amount, project_id)

    def update_project_status(self, project_id: str, status: str):
        """Update the status of a client project."""
        proj = self.client_projects.get(project_id)
        if not proj:
            logging.warning("Client project %s not found.", project_id)
            return
        proj.status = status
        if status == "delivered":
            proj.delivered_at = datetime.now(timezone.utc).isoformat()
        self._save_data()

    def get_revenue_summary(self) -> Dict[str, Any]:
        """Calculate revenue statistics."""
        total_revenue = sum(entry.get("amount", 0) for entry in self.revenue_log)
        total_projects = len(self.client_projects)
        active_projects = sum(
            1
            for p in self.client_projects.values()
            if p.status in ("accepted", "in_progress")
        )
        pending_proposals = sum(
            1 for p in self.client_projects.values() if p.status == "proposal"
        )
        pipeline_value = sum(
            p.price
            for p in self.client_projects.values()
            if p.status in ("proposal", "accepted", "in_progress")
        )

        return {
            "total_revenue": total_revenue,
            "total_projects": total_projects,
            "active_projects": active_projects,
            "pending_proposals": pending_proposals,
            "pipeline_value": pipeline_value,
            "payments": len(self.revenue_log),
        }

    async def identify_opportunities(self) -> Dict[str, Any]:
        """
        Use LLM to analyze the business and suggest revenue opportunities.
        """
        stats = self.get_revenue_summary()

        # Gather skills and capabilities from memory
        capabilities = ""
        if self.memory:
            skills = self.memory.get_by_category("skills", limit=10)
            if skills:
                capabilities = "Known skills:\n" + "\n".join(
                    f"- {s.content}" for s in skills
                )

        prompt = f"""You are a business strategist for an AI-powered software development company.

Current business metrics:
{json.dumps(stats, indent=2)}

{capabilities}

Identify opportunities to generate revenue. Target: $25,000/month.

Generate a JSON object with:
- "opportunities": List of revenue opportunities, each with "name", "description", "estimated_monthly_revenue", "effort_level"
- "quick_wins": Immediate actions to take
- "long_term_strategy": Strategic recommendations
- "monthly_target_plan": How to reach $25k/month

Respond ONLY with a JSON object."""

        try:
            response = await ollama.AsyncClient().chat(
                model=config.COGNITIVE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
            return json.loads(response["message"]["content"])
        except (json.JSONDecodeError, Exception) as exc:
            logging.error("Failed to identify opportunities: %s", exc)
            return {
                "opportunities": [],
                "quick_wins": [],
                "long_term_strategy": f"Analysis failed: {exc}",
                "monthly_target_plan": "",
            }
