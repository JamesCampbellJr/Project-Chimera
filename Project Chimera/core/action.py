# /core/action.py

import pyautogui
import subprocess
import logging
import time
from typing import List, Dict, Any

# Disable pyautogui failsafe
pyautogui.FAILSAFE = False

class Action:
    """
    Executes actions on the PC based on a structured plan.
    """
    def __init__(self):
        pyautogui.PAUSE = 0.5  # A small pause after each action
        logging.info("Action module initialized.")

    def execute_plan(self, plan: List[Dict[str, Any]]):
        """
        Iterates through a plan and executes each action.
        Note: This is a synchronous function that will block.
        The agent's run loop should `await` this in an executor.
        """
        logging.info(f"Executing plan: {plan}")
        for step in plan:
            action_type = step.get("action")
            try:
                if action_type == "type_text":
                    self._type_text(step.get("text", ""))
                elif action_type == "press_key":
                    self._press_key(step.get("key"))
                elif action_type == "click":
                    self._click(step.get("button", "left"))
                elif action_type == "double_click":
                    self._double_click()
                elif action_type == "scroll":
                    self._scroll(step.get("direction", "down"), step.get("amount", 100))
                elif action_type == "run_command":
                    self._run_command(step.get("command"))
                elif action_type == "wait":
                    self._wait(step.get("seconds", 1))
                # The actions below are special and handled by the agent loop, not here.
                elif action_type in ["capture_and_analyze_screen", "spawn_agent", "finish_task", "move_mouse_to_element"]:
                    logging.info(f"Control action '{action_type}' received. Will be handled by the agent.")
                    # We just break the local execution and return the control action
                    return step 
                else:
                    logging.warning(f"Unknown action type: {action_type}")
            except Exception as e:
                logging.error(f"Error executing action {step}: {e}")
                # Return a special error action
                return {"action": "finish_task", "reason": f"Error during execution of action '{action_type}'."}
        
        # If the plan completes without a control action, we are done.
        return {"action": "finish_task", "reason": "Plan executed successfully."}


    def _type_text(self, text: str):
        logging.info(f"Typing text: '{text}'")
        pyautogui.write(text, interval=0.05)

    def _press_key(self, key: str):
        logging.info(f"Pressing key: '{key}'")
        pyautogui.press(key)

    def _click(self, button: str = 'left'):
        logging.info(f"Clicking {button} mouse button.")
        pyautogui.click(button=button)
        
    def _double_click(self):
        logging.info("Double clicking left mouse button.")
        pyautogui.doubleClick()

    def _scroll(self, direction: str, amount: int):
        logging.info(f"Scrolling {direction} by {amount} units.")
        scroll_amount = -amount if direction == 'down' else amount
        pyautogui.scroll(scroll_amount)

    def _run_command(self, command: str):
        logging.info(f"Running shell command: '{command}'")
        # Use subprocess.Popen for non-blocking execution
        subprocess.Popen(command, shell=True)
        
    def _wait(self, seconds: float):
        logging.info(f"Waiting for {seconds} seconds.")
        time.sleep(seconds)

    # Note: move_mouse_to_element is handled in the agent class as it needs access to screen analysis results
    def move_mouse(self, x: int, y: int):
        logging.info(f"Moving mouse to ({x}, {y})")
        pyautogui.moveTo(x, y, duration=0.25)