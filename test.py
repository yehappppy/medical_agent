from agent.utils.tools import logger, get_agent

agent = get_agent()
logger.info("Agent initialized successfully.")
agent_response = agent.invoke("Hello, how can I assist you today?")
logger.info(f"Agent response: {agent_response.content}")
logger.info("Test completed.")
