# /orchestrator.py

import asyncio
import logging
from typing import Dict

from agent import Agent, PrometheusAgent
from tutor_agent import AthenaAgent

class Orchestrator:
    """
    Manages the lifecycle of all agents, routes tasks, and facilitates communication.
    """
    def __init__(self):
        self.agent_registry: Dict[str, Agent] = {}
        self.task_queue = asyncio.Queue()
        self.running_tasks = []
        self.known_skills = {} # Maps skill name to file path
        logging.info("Orchestrator initialized.")
        # TODO: Replace with a network-based message bus (e.g., RabbitMQ)
        self.message_bus = asyncio.Queue()

    async def start(self):
        """Starts the orchestrator and its primary agent(s)."""
        logging.info("Orchestrator starting...")
        
        # Spawn the primary agent, Prometheus
        prometheus = PrometheusAgent(self)
        self.agent_registry[prometheus.id] = prometheus
        
        # Start the agent's main loop as a background task
        prometheus_task = asyncio.create_task(prometheus.run())
        self.running_tasks.append(prometheus_task)
        
        # Wait for all tasks to complete (e.g., on shutdown)
        await asyncio.gather(*self.running_tasks)
        logging.info("Orchestrator has shut down.")

    async def spawn_agent(self, role: str, task: str):
        """Spawns a new agent for a specific role and task."""
        logging.info(f"Spawning agent for role '{role}' with task: '{task}'")
        
        if role.lower() == 'tutor':
            new_agent = AthenaAgent(self, task_to_learn=task)
        else:
            # Placeholder for other specialized agents
            logging.warning(f"No specific agent class for role '{role}'. Using base Agent.")
            new_agent = Agent(self, role=role)

        self.agent_registry[new_agent.id] = new_agent
        
        # Start the new agent's run loop and add its task
        agent_task = asyncio.create_task(new_agent.run())
        await new_agent.mailbox.put(task)
        self.running_tasks.append(agent_task)

    async def report_skill_learned(self, skill_name: str, skill_path: str):
        """Callback for agents to report a new skill."""
        logging.info(f"New skill learned: '{skill_name}'. Stored at: {skill_path}")
        self.known_skills[skill_name] = skill_path
        # TODO: Propagate this new skill to other agents or a central knowledge base.
        # For now, we just log it.
        
    # --- Inter-Agent Communication (Simple Pub/Sub) ---
    async def publish_message(self, message: dict):
        """Publishes a message to the central bus."""
        # TODO: Replace with RabbitMQ or Redis pub/sub call
        # e.g., channel.basic_publish(exchange='chimera_bus', routing_key='some.topic', body=json.dumps(message))
        await self.message_bus.put(message)
        
    async def message_listener(self):
        """Listens to the message bus and routes messages."""
        while True:
            message = await self.message_bus.get()
            # TODO: Implement routing logic based on message content/topic
            logging.info(f"Message received on bus: {message}")
            self.message_bus.task_done()