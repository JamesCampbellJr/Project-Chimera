# /agents/business_agent.py

"""
Business Development Agent — Manages revenue generation and client relations.

Works with the BusinessManager to generate proposals, track revenue,
and identify opportunities for growth towards the $25k/month target.
"""

import asyncio
import logging
import os
from typing import Optional

from agent import Agent
from core.business_manager import BusinessManager
import config


class BusinessAgent(Agent):
    """
    Specialist agent focused on business development and revenue generation.
    """

    def __init__(self, orchestrator, memory_store=None):
        super().__init__(
            orchestrator,
            role=config.BUSINESS_ROLE,
            agent_id=f"BusinessDev-{os.getpid()}",
        )
        self.memory = memory_store
        self.business_manager = BusinessManager(memory_store=memory_store)

    async def process_task(self, task: str):
        """
        Process business-related tasks:
        - Generate proposals
        - Analyze revenue
        - Identify opportunities
        """
        logging.info("BusinessAgent processing: %s", task)
        task_lower = task.lower()

        try:
            if "proposal" in task_lower:
                await self._handle_proposal(task)
            elif "revenue" in task_lower or "report" in task_lower:
                await self._handle_revenue_report()
            elif "opportunity" in task_lower or "growth" in task_lower:
                await self._handle_opportunities()
            else:
                # Default: identify opportunities
                await self._handle_opportunities()

        except Exception as exc:
            logging.error("BusinessAgent error: %s", exc, exc_info=True)

    async def _handle_proposal(self, task: str):
        """Generate a client proposal."""
        # Extract client info from task (simplified)
        proposal = await self.business_manager.generate_proposal(
            client_name="Prospective Client",
            project_description=task,
        )
        logging.info("Proposal generated: %s", proposal.get("title", "N/A"))

        if self.memory:
            self.memory.store(
                content=f"Generated proposal: {proposal.get('title', task)}. Price: ${proposal.get('price_usd', 0)}",
                category="business",
                tags=["proposal_generated"],
                importance=0.7,
            )

    async def _handle_revenue_report(self):
        """Generate and log a revenue report."""
        summary = self.business_manager.get_revenue_summary()
        logging.info("Revenue Summary: %s", summary)

        if self.memory:
            self.memory.store(
                content=(
                    f"Revenue report: Total=${summary['total_revenue']}, "
                    f"Active={summary['active_projects']}, "
                    f"Pipeline=${summary['pipeline_value']}"
                ),
                category="business",
                tags=["revenue_report"],
                importance=0.6,
            )

    async def _handle_opportunities(self):
        """Identify and log business opportunities."""
        opportunities = await self.business_manager.identify_opportunities()
        logging.info("Opportunities identified: %s", opportunities.get("opportunities", []))

        if self.memory:
            opps = opportunities.get("opportunities", [])
            self.memory.store(
                content=f"Identified {len(opps)} business opportunities. Quick wins: {opportunities.get('quick_wins', [])}",
                category="business",
                tags=["opportunities"],
                importance=0.8,
            )
