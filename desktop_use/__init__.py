from logging_config import setup_logging


setup_logging()

# from agent.prompts import SystemPrompt as SystemPrompt
# from agent.service import Agent as Agent
# from agent.views import ActionModel as ActionModel
# from agent.views import ActionResult as ActionResult
# from agent.views import AgentHistoryList as AgentHistoryList
# from desktop.desktop import DesktopConfig as DesktopConfig
# from desktop.context import DesktopContextConfig as DesktopContextConfig
# from desktop.context import DesktopContext as DesktopContext
# from controller.service import Controller as Controller
# from dom.service import DesktopDomService as DesktopDomService

# __all__ = [
# 	'Agent',
# 	'DesktopContext',
# 	'DesktopConfig',
# 	'Controller',
# 	'DesktopDomService',
# 	'SystemPrompt',
# 	'ActionResult',
# 	'ActionModel',
# 	'AgentHistoryList',
# 	'DesktopContextConfig',
# ]
from logging_config import setup_logging

setup_logging()

from agent.prompts import SystemPrompt as SystemPrompt
from agent.service import Agent as Agent
from agent.views import ActionModel as ActionModel
from agent.views import ActionResult as ActionResult
from agent.views import AgentHistoryList as AgentHistoryList
from desktop.desktop import Desktop as Desktop
from desktop.desktop import DesktopConfig as DesktopConfig
from desktop.context import DesktopContext as DesktopContext
from desktop.context import DesktopContextConfig as DesktopContextConfig
from controller.service import Controller as Controller


__all__ = [
    'Agent',
    'DesktopContext',
    'Desktop',
    'DesktopConfig',
    'Controller',
    'ElementService',
    'SystemPrompt',
    'ActionResult',
    'ActionModel',
    'AgentHistoryList',
    'DesktopContextConfig',
]