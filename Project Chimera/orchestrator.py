# /orchestrator.py

import asyncio
import logging
from typing import Dict

from agent import Agent, PrometheusAgent
from core.tutor_agent import AthenaAgent
from core.memory import MemoryStore
from core.self_improvement import SelfImprovement
from agents.builder_agent import BuilderAgent
from agents.business_agent import BusinessAgent
from agents.review_agent import ReviewAgent

class Orchestrator:
    """
    Manages the lifecycle of all agents, routes tasks, and facilitates communication.
    Integrates the memory system for persistent knowledge and self-improvement.
    """
    def __init__(self):
        self.agent_registry: Dict[str, Agent] = {}
        self.task_queue = asyncio.Queue()
        self.running_tasks = []
        self.known_skills = {} # Maps skill name to file path
        self.memory = MemoryStore()
        self.self_improvement = SelfImprovement(memory_store=self.memory)
        logging.info("Orchestrator initialized with memory and self-improvement.")
        self.message_bus = asyncio.Queue()

    async def start(self):
        """Starts the orchestrator and its primary agent(s)."""
        logging.info("Orchestrator starting...")
        
        # Log memory stats on startup
        stats = self.memory.get_stats()
        if stats.get("total", 0) > 0:
            logging.info("Loaded %d memories from previous sessions.", stats["total"])
        
        # Spawn the primary agent, Prometheus
        prometheus = PrometheusAgent(self)
        self.agent_registry[prometheus.id] = prometheus
        
        # Start the agent's main loop as a background task
        prometheus_task = asyncio.create_task(prometheus.run())
        self.running_tasks.append(prometheus_task)
        
        # Start the message listener
        listener_task = asyncio.create_task(self.message_listener())
        self.running_tasks.append(listener_task)
        
        # Wait for all tasks to complete (e.g., on shutdown)
        await asyncio.gather(*self.running_tasks, return_exceptions=True)
        logging.info("Orchestrator has shut down.")

    async def spawn_agent(self, role: str, task: str):
        """Spawns a new agent for a specific role and task."""
        logging.info(f"Spawning agent for role '{role}' with task: '{task}'")
        
        role_lower = role.lower()
        
        if role_lower == 'tutor':
            new_agent = AthenaAgent(self, task_to_learn=task)
        elif role_lower in ('builder', 'developer', 'engineer', 'software engineer'):
            new_agent = BuilderAgent(
                self, project_description=task, memory_store=self.memory
            )
        elif role_lower in ('business', 'business development', 'sales'):
            new_agent = BusinessAgent(self, memory_store=self.memory)
        elif role_lower in ('reviewer', 'code reviewer', 'qa'):
            new_agent = ReviewAgent(self, memory_store=self.memory)
        else:
            logging.warning(f"No specific agent class for role '{role}'. Using base Agent.")
            new_agent = Agent(self, role=role)

        self.agent_registry[new_agent.id] = new_agent
        
        # Start the new agent's run loop and add its task
        agent_task = asyncio.create_task(new_agent.run())
        await new_agent.mailbox.put(task)
        self.running_tasks.append(agent_task)
        
        # Record the spawn event in memory
        self.memory.store(
            content=f"Spawned {role} agent for task: {task}",
            category="general",
            tags=["agent_spawn", role_lower],
            importance=0.5,
        )

    async def report_skill_learned(self, skill_name: str, skill_path: str):
        """Callback for agents to report a new skill."""
        logging.info(f"New skill learned: '{skill_name}'. Stored at: {skill_path}")
        self.known_skills[skill_name] = skill_path
        
        # Persist skill to memory for future reference
        self.memory.store(
            content=f"Skill learned: {skill_name}. Location: {skill_path}",
            category="skills",
            tags=["learned_skill", skill_name.replace(" ", "_")],
            importance=0.8,
        )
        
    # --- Inter-Agent Communication (Simple Pub/Sub) ---
    async def publish_message(self, message: dict):
        """Publishes a message to the central bus."""
        await self.message_bus.put(message)
        
    async def message_listener(self):
        """Listens to the message bus and routes messages."""
        while True:
            message = await self.message_bus.get()
            logging.info(f"Message received on bus: {message}")
            
            # Route messages to target agents if specified
            target_id = message.get("target_agent")
            if target_id and target_id in self.agent_registry:
                agent = self.agent_registry[target_id]
                await agent.mailbox.put(message.get("content", ""))
            
            self.message_bus.task_done()