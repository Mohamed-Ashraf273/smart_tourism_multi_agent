import json

from typing import Any
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate

class TripBudgetPlannerTool(BaseTool):
    name: str = "trip_budget_planner"
    description: str = "Calculate trip cost. Input: any text about trip"

    llm: Any = None
    parser_prompt: Any = None
    chain: Any = None
    
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        self.parser_prompt = ChatPromptTemplate.from_template(
            "Extract city, days, and style from: {user_input}. Respond ONLY with JSON: {{'city': 'Dubai', 'days': 3, 'style': 'standard'}}"
        )
        self.chain = self.parser_prompt | self.llm
    
    def _run(self, user_input: str) -> str:
        response = self.chain.invoke({"user_input": user_input})
        data = {"city": "Dubai", "days": 3, "style": "standard"}
        
        try:
            json_str = response.content.strip()
            if "{" in json_str and "}" in json_str:
                json_str = json_str[json_str.index("{"):json_str.rindex("}")+1]
                parsed = json.loads(json_str)
                data.update(parsed)
        except:
            pass
        
        prices = {"budget": 150, "standard": 400, "luxury": 1000}
        total = data["days"] * prices.get(data["style"], 400)
        
        return f"{data['days']} day {data['style']} trip to {data['city']}: {total} AED"