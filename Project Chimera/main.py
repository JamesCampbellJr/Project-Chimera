 # /main.py

import asyncio
import logging
from orchestrator import Orchestrator
import config

# Configure logging
logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

async def main():
    """
    Initializes and starts the Project Chimera system.
    """
    logging.info("Initializing Project Chimera...")
    
    # Create and start the central orchestrator
    chimera_orchestrator = Orchestrator()
    
    # The orchestrator will spawn the primary agent (Prometheus) and manage the system.
    # It will run until explicitly stopped.
    await chimera_orchestrator.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Project Chimera shutting down.")
    except Exception as e:
        logging.critical(f"A critical error occurred: {e}", exc_info=True)