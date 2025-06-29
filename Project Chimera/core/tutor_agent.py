# /tutor_agent.py
import asyncio
import logging
import os
import json

from agent import Agent
from core.perception import Perception
from core.cognition import Cognition
from core.action import Action

class AthenaAgent(Agent):
    """
    A 'Tutor' agent that learns new skills.
    It researches a topic online, synthesizes the knowledge, and saves it
    as a new skill (a reusable plan or script).
    """
    def __init__(self, orchestrator, task_to_learn: str):
        super().__init__(orchestrator, role="Tutor", agent_id=f"Athena-{os.getpid()}")
        self.perception = Perception()
        self.cognition = Cognition()
        self.action = Action()
        self.task_to_learn = task_to_learn
        self.research_data = ""
        self.last_screen_analysis = None

    async def process_task(self, task):
        """The learning process: Research -> Synthesize -> Save Skill."""
        logging.info(f"Athena starting to learn: {self.task_to_learn}")
        
        # 1. Research Phase
        await self._research_phase()
        
        # 2. Synthesize Phase
        new_skill = await self._synthesize_phase()
        
        # 3. Save Skill
        self._save_skill(new_skill)

        # 4. Report back to orchestrator
        await self.orchestrator.report_skill_learned(self.task_to_learn, f"skills/{self.task_to_learn.replace(' ', '_')}.json")
        logging.info(f"Athena finished learning: {self.task_to_learn}. Shutting down.")
        await self.stop()

    async def _research_phase(self):
        """Uses PC control to browse the web and gather information."""
        logging.info("Starting research phase...")
        # This is a simplified plan. A real implementation would be more robust.
        research_plan = [
            {"action": "run_command", "command": "chrome.exe https://www.google.com"},
            {"action": "wait", "seconds": 3},
            {"action": "type_text", "text": f"how to {self.task_to_learn} step by step"},
            {"action": "press_key", "key": "enter"},
            {"action": "wait", "seconds": 3},
            # In a real scenario, this would be a loop:
            # - Analyze screen to find and click links
            # - Scrape text from pages
            # - Add scraped text to self.research_data
            # For this example, we'll assume the first page has the info.
            {"action": "finish_task", "reason": "Research complete."}
        ]
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.action.execute_plan, research_plan)
        # TODO: Implement actual web scraping and data collection.
        # For now, we'll use placeholder data.
        self.research_data = f"Placeholder research data for '{self.task_to_learn}'. A real agent would have scraped tutorials and articles from the web."
        logging.info("Research phase complete.")

    async def _synthesize_phase(self) -> dict:
        """Uses the LLM to process research data and create a new skill plan."""
        logging.info("Starting synthesis phase...")
        synthesis_prompt = f"""
        Based on the following research data, create a generic, reusable JSON action plan
        to accomplish the task: "{self.task_to_learn}".
        
        The plan should use the available actions (type_text, press_key, etc.) and be
        as general as possible so it can be reused later. Use placeholders like
        <PARAMETER_1>, <PARAMETER_2> if the task requires specific input.
        
        Research Data:
        ---
        {self.research_data}
        ---
        
        Your response must be a single JSON object containing two keys: "description" and "plan".
        - "description": A short explanation of what this skill does.
        - "plan": The JSON array of action objects.
        """
        
        response = await self.cognition.cognition_client.chat(
            model=config.COGNITIVE_MODEL,
            messages=[{'role': 'user', 'content': synthesis_prompt}],
            format='json'
        )
        try:
            skill_data = json.loads(response['message']['content'])
            logging.info("Successfully synthesized new skill.")
            return skill_data
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Failed to synthesize skill: {e}")
            return {"description": "Synthesis failed.", "plan": []}

    def _save_skill(self, skill_data: dict):
        """Saves the learned skill to the 'skills' directory."""
        if not os.path.exists('skills'):
            os.makedirs('skills')
            
        skill_name = self.task_to_learn.replace(' ', '_').lower()
        file_path = f"skills/{skill_name}.json"
        
        with open(file_path, 'w') as f:
            json.dump(skill_data, f, indent=4)
        logging.info(f"New skill '{skill_name}' saved to {file_path}")

        # TODO: Implement the advanced fine-tuning trigger
        # self.trigger_finetuning(skill_data)

    def trigger_finetuning(self, skill_data):
        logging.info("Placeholder for LoRA fine-tuning process.")
        # 1. Prepare a dataset from self.research_data and skill_data.
        #    Format: [{"instruction": "How do I do X?", "input": "context", "output": "step-by-step plan"}]
        # 2. Call an external script (e.g., using `subprocess`) that runs `axolotl` or a custom PyTorch fine-tuning loop.
        # 3. The script would train a LoRA adapter on the base model.
        # 4. The orchestrator would need to be notified that a new adapter is available.