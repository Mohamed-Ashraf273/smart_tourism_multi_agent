import json

from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate
from typing import Any

class PrayerTimerTool(BaseTool):
    name: str = "prayer_timer"
    description: str = "Get prayer times for any city and date using Aladhan API"

    llm: Any = None
    get_tool: Any = None
    tool2_prompt: Any = None
    chain: Any = None

    def __init__(self, llm: Any, get_tool: Any):
        super().__init__()
        self.llm = llm
        self.get_tool = get_tool
        self.tool2_prompt = ChatPromptTemplate.from_template(
            "Given the user input: {input} write a valid URL. "
            "Without adding comments or markdown or anything just provide the url\n"
            "URL: https://api.aladhan.com/v1/timingsByAddress/dd-mm-yyyy?address=City,Country&method=8"
        )
        self.chain = self.tool2_prompt | self.llm

    def _run(self, user_input: str):
        try:
            url_response = self.chain.invoke({"input": user_input})
            api_url = url_response.content.strip()
            api_response = self.get_tool.run(api_url)
            timings = json.loads(api_response)['data']['timings']
            return timings
        except Exception as e:
            return f"Error getting prayer times: {str(e)}"
