# /core/cognition.py

import logging
import ollama
import json
from typing import List, Dict, Any

import config

class Cognition:
    """
    The cognitive core of an agent. It uses an LLM to process information,
    reason about it, and generate a plan of action.
    """
    def __init__(self):
        logging.info("Cognition module initialized.")

    async def generate_plan(self, user_prompt: str, screen_analysis: dict, agent_role: str) -> List[Dict[str, Any]]:
        """
        Takes user prompt and screen analysis, and generates an actionable plan.
        """
        logging.info("Generating a plan of action...")
        
        context = f"""
        You are an autonomous AI agent controlling a Windows PC to assist a user.
        Your current role is: "{agent_role}"

        The user's command is: "{user_prompt}"

        Here is an analysis of the current screen content:
        {json.dumps(screen_analysis, indent=2)}

        Your task is to create a step-by-step plan to fulfill the user's request.
        Your response MUST be a JSON array of action objects. Do not add any explanation or other text.
        Each object in the array represents one action.

        Available actions are:
        - {{ "action": "type_text", "text": "text to type" }}
        - {{ "action": "press_key", "key": "key name (e.g., 'enter', 'ctrl', 'f5')" }}
        - {{ "action": "move_mouse_to_element", "element_id": <id_from_screen_analysis> }}  (This is PREFERRED for clicking)
        - {{ "action": "click", "button": "left/right" }} (Use after moving the mouse)
        - {{ "action": "double_click" }}
        - {{ "action": "scroll", "direction": "up/down", "amount": <pixels> }}
        - {{ "action": "run_command", "command": "shell command to execute" }} (e.g., "notepad.exe")
        - {{ "action": "wait", "seconds": <number> }}
        - {{ "action": "capture_and_analyze_screen" }} (To re-evaluate the screen after an action)
        - {{ "action": "spawn_agent", "role": "<role_name>", "task": "<description_of_task>" }} (To delegate a complex task, e.g., 'Web Researcher', 'Tutor')
        - {{ "action": "finish_task", "reason": "A summary of why the task is complete." }}

        Think step-by-step. If the screen analysis provides an element_id, use "move_mouse_to_element". 
        If not, you may need to use `run_command` to open an app, or `capture_and_analyze_screen` to get more context.
        If the task is complex and requires learning (e.g., "learn how to use this new software"), use `spawn_agent` with the 'Tutor' role.

        Example Plan:
        [
            {{ "action": "run_command", "command": "chrome.exe https://www.google.com" }},
            {{ "action": "wait", "seconds": 3 }},
            {{ "action": "capture_and_analyze_screen" }},
            {{ "action": "type_text", "text": "latest AI news" }},
            {{ "action": "press_key", "key": "enter" }}
        ]

        Now, generate the plan for the user's request.
        """
        
        try:
            response = await ollama.AsyncClient().chat(
                model=config.COGNITIVE_MODEL,
                messages=[{'role': 'user', 'content': context}],
                format='json'
            )
            response_text = response['message']['content']
            logging.debug(f"Raw LLM plan response: {response_text}")
            plan = json.loads(response_text)
            if isinstance(plan, list):
                return plan
            else:
                logging.error(f"LLM did not return a list. Response: {plan}")
                return [{"action": "finish_task", "reason": "Error: Plan generation failed."}]
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from LLM plan response. Raw: {response_text}")
            return [{"action": "finish_task", "reason": "Error: Could not decode the plan from the LLM."}]
        except Exception as e:
            logging.error(f"An error occurred during plan generation: {e}")
            return [{"action": "finish_task", "reason": f"Error: An exception occurred: {e}"}]