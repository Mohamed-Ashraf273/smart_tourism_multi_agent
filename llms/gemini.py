from langchain_google_genai import ChatGoogleGenerativeAI

class Gemini:
    def __init__(self, api_key, temperature=0.7):
        self.api_key = api_key
        self.model = ChatGoogleGenerativeAI(
            model='gemini-2.5-flash-lite',
            api_key=api_key,
            temperature=temperature
        )

    def __call__(self, prompt):
        return self.model.invoke(prompt)
    
    def llm(self):
        return self.model