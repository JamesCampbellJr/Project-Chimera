# /agent.py

import asyncio
import logging
import uuid
from typing import Optional

from core.voice_interface import VoiceInterface
from core.perception import Perception
from core.cognition import Cognition
from core.action import Action
import config

class Agent:
    """Base class for all agents in the system."""
    def __init__(self, orchestrator, role: str, agent_id: Optional[str] = None):
        self.id = agent_id or str(uuid.uuid4())
        self.role = role
        self.orchestrator = orchestrator
        self.mailbox = asyncio.Queue()
        self.is_running = False
        self.current_task = None
        logging.info(f"Agent {self.id} ({self.role}) created.")

    async def run(self):
        """Main loop for the agent."""
        self.is_running = True
        while self.is_running:
            try:
                task = await self.mailbox.get()
                self.current_task = task
                logging.info(f"Agent {self.id} received task: {task}")
                await self.process_task(task)
                self.current_task = None
                self.mailbox.task_done()
            except asyncio.CancelledError:
                self.is_running = False
                logging.info(f"Agent {self.id} run loop cancelled.")
                break
        logging.info(f"Agent {self.id} has shut down.")

    async def process_task(self, task):
        """Placeholder for task processing logic. To be implemented by subclasses."""
        raise NotImplementedError

    async def stop(self):
        """Stops the agent's main loop."""
        self.is_running = False
        # Put a dummy item in the queue to unblock the `await self.mailbox.get()`
        await self.mailbox.put(None)


class PrometheusAgent(Agent):
    """The primary user-facing agent."""
    def __init__(self, orchestrator):
        super().__init__(orchestrator, role=config.PROMETHEUS_ROLE, agent_id="Prometheus")
        self.voice = VoiceInterface()
        self.perception = Perception()
        self.cognition = Cognition()
        self.action = Action()
        self.last_screen_analysis = None
        
    async def run(self):
        """Prometheus's main loop: listen for commands and execute them."""
        self.is_running = True
        self.voice.speak("Prometheus is online.")
        
        while self.is_running:
            try:
                # 1. Listen for user command
                user_command = await self.voice.listen_for_hotword_and_command()
                if not user_command:
                    continue

                await self.process_task(user_command)

            except asyncio.CancelledError:
                self.is_running = False
                break
            except Exception as e:
                logging.error(f"Error in Prometheus main loop: {e}", exc_info=True)
                self.voice.speak("I've encountered an error. Please check the logs.")
                await asyncio.sleep(2)

    async def process_task(self, user_command: str):
        """Processes a user command by executing a loop of perception, cognition, and action."""
        logging.info(f"Prometheus processing task: '{user_command}'")
        self.voice.speak(f"Okay, working on: {user_command}")
        
        # Initial screen analysis
        await self._capture_and_analyze_screen(user_command)

        loop_count = 0
        max_loops = 10 # Safety break
        
        while loop_count < max_loops:
            loop_count += 1
            
            # 2. Cognition: Generate a plan
            plan = await self.cognition.generate_plan(user_command, self.last_screen_analysis, self.role)
            
            if not plan:
                self.voice.speak("I'm having trouble forming a plan. I'll stop here.")
                break
                
            # 3. Action: Execute the first step of the plan
            # The action executor returns a control action if it encounters one
            loop = asyncio.get_running_loop()
            control_action = await loop.run_in_executor(
                None, self.action.execute_plan, plan
            )

            # 4. Handle control actions
            action_type = control_action.get("action")
            
            if action_type == "finish_task":
                reason = control_action.get('reason', 'Task completed.')
                logging.info(f"Task finished. Reason: {reason}")
                self.voice.speak(reason)
                break
            
            elif action_type == "capture_and_analyze_screen":
                logging.info("Re-evaluating screen as per plan.")
                await self._capture_and_analyze_screen(user_command)
                continue # Go to the next loop iteration to form a new plan
                
            elif action_type == "spawn_agent":
                role = control_action.get('role')
                task = control_action.get('task')
                logging.info(f"Requesting orchestrator to spawn '{role}' agent for task: {task}")
                # TODO: Implement response handling from the spawned agent
                await self.orchestrator.spawn_agent(role, task)
                self.voice.speak(f"I've delegated the task of '{task}' to a specialist agent.")
                break # End current task, as it's been delegated
                
            elif action_type == "move_mouse_to_element":
                element_id = control_action.get('element_id')
                # This needs access to the screen analysis data
                self._move_to_element_by_id(element_id)
                # This action is usually followed by a click, so we continue execution in the next loop
                continue

        if loop_count >= max_loops:
            logging.warning("Reached max loop iteration. Task may be stuck.")
            self.voice.speak("I seem to be stuck in a loop. I'm stopping this task for safety.")

    async def _capture_and_analyze_screen(self, user_command):
        loop = asyncio.get_running_loop()
        # Run blocking IO in executor
        screenshot = await loop.run_in_executor(None, self.perception.capture_screen)
        self.last_screen_analysis = await self.perception.analyze_screen(screenshot, user_command)
        logging.info(f"Screen analysis complete: {self.last_screen_analysis.get('description')}")
        
    def _move_to_element_by_id(self, element_id: int):
        """Finds an element by its ID in the last screen analysis and moves the mouse to it."""
        if not self.last_screen_analysis or 'elements' not in self.last_screen_analysis:
            logging.warning("Cannot move to element: No screen analysis available.")
            return

        element_to_find = next((el for el in self.last_screen_analysis['elements'] if el.get('id') == element_id), None)

        if not element_to_find:
            logging.warning(f"Could not find element with ID {element_id} in the analysis.")
            return

        # VLM doesn't provide coordinates, so this is a placeholder.
        # A more advanced VLM or a different approach (e.g., GPT-4o with screen drawing) would be needed.
        # For now, we simulate by logging. A future implementation would parse 'coordinates'.
        logging.warning(f"ACTION REQUIRED: Human intervention or a more advanced VLM needed to find coordinates for element: {element_to_find}")
        self.voice.speak(f"I need to interact with the element labeled '{element_to_find.get('label')}', but I can't determine its exact location yet.")