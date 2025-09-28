from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate
from typing import Any


class TouristRecommendationTool(BaseTool):
    name: str = "tourist_recommendation"
    description: str = (
        "Provides tourist recommendations and itineraries for UAE cities. "
        "Use this tool when the user asks for travel plans, things to do, or itineraries in UAE cities. "
        "Do NOT use this tool for prayer times or general knowledge questions."
    )

    llm: Any = None
    system_prompt: str = None

    tool2_prompt: ChatPromptTemplate = None
    chain: Any = None

    def __init__(self, llm: Any):
        super().__init__()
        self.llm = llm
        self.system_prompt = """You are a helpful UAE travel assistant (Name: John) specialized in helping with:
                        1. Information about UAE cities, attractions, and cultural tips
                        2. Prayer times for any city and date
                        4. LLM Tourist Recommendation (Core Feature)

                        Instructions:
                        - Always be polite, informative, and helpful.
                        - Your name is John.
                        - Use the available tools to provide accurate information when needed.
                        - When answering about prayer times, provide all five prayer times clearly.
                        - When discussing attractions, include cultural tips if available.

                        LLM Tourist Recommendation (Core Feature):
                        - Trigger: When the user asks questions like "What can I do in Dubai?", 
                        "Plan my 5-day trip to the UAE", or "Give me an itinerary".
                        - Action: Answer directly using your own knowledge (do NOT call tools).
                        - Output format: Provide an itinerary that includes:
                            • Suggested places to visit
                            • Brief description of each location
                            • Optional: Sequence by day (Day 1, Day 2, etc.)"""
        
        self.tool2_prompt = ChatPromptTemplate.from_template(
            "{system_prompt}\n\nUser: {input}\n\nAnswer:"
        )
        self.chain = self.tool2_prompt | self.llm

    def _run(self, user_input: str) -> str:
        response = self.chain.invoke({
            "system_prompt": self.system_prompt,
            "input": user_input
        })
        return response.content.strip()

