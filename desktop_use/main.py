
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from controller.service import Controller
from desktop.desktop import Desktop
from desktop.context import DesktopContext
from agent.service import Agent
from agent.prompts import SystemPrompt

load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
        
        llm = ChatOpenAI(
            model='gpt-4o',
            temperature=0.0,
        )
        
       
        desktop = Desktop()
        desktop_context = DesktopContext(desktop=desktop, config=desktop.config.new_context_config)
        controller = Controller()

        
        
      
        task = "open notepad and write a paragraph about the importance of AI in modern technology."  
       
        logger.info(f"Task: {task}")  


        
        agent = Agent(
            task=task,
            llm=llm,
            controller=controller,
            desktop=desktop,
            desktop_context=desktop_context,
            use_screenshot=True,  
            system_prompt_class=SystemPrompt,
            max_failures=3
        )
        
        
        history = await agent.run(max_steps=5)
        
        

if __name__ == "__main__":
    asyncio.run(main())