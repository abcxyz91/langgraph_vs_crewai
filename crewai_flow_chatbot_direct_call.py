from crewai.flow.flow import Flow, listen, router, start
from crewai import LLM
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from typing import Literal, List, Dict
import os, json

# Load environment variables - Same as langgraph
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("MODEL")
os.environ["CREWAI_DISABLE_TELEMETRY"] = "True" # Disable telemetry error message in terminal

# Define LLM model - Used CrewAI's `LLM` class
gemini_llm = LLM(
    model=GEMINI_MODEL,
    temperature=0,
    max_tokens=4096
)

# Define the state model - Used Pydantic `BaseModel` for structured state instead of TypedDict
# Langgraph's add_message automatically append conversation history. 
# CrewAI can even maintain conversation history between sessions by @persist decorator
# In this example, to maintain coversation history within session, need to do it manually
class MessageState(BaseModel):
    user_message: str = ""
    message_type: str = ""
    response: str = ""
    conversation_history: List[Dict[str, str]] = [] # Save conversation history

# Define the classifier model - Same as langgraph
class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response."
    )

class ChatbotFlow(Flow[MessageState]): # Replaced StateGraph with CrewAI Flow class. This Flow receive MessageState class as structure State
    # Used `@start()`, `@router()`, and `@listen()` instead of graph nodes and edges
    @start()
    def classify_message(self):
        # Get user input from state
        if not self.state.user_message:
            return "exit"
        
        # Used CrewAI's `LLM` class with structured output via `response_format`
        classifier_llm = LLM(
            model=GEMINI_MODEL,
            response_format=MessageClassifier
        )
        
        result = classifier_llm.call([
            {
                "role": "system",
                "content": """Classify the user message as either:
                - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
                - 'logical': if it asks for facts, information, logical analysis, or practical solutions
                """
            },
            {"role": "user", "content": self.state.user_message}
        ])
        
        try:
            # LLM response could be a dict or a string, hence the parsing will handle both case
            data = result if isinstance(result, dict) else json.loads(result)
            parsed = MessageClassifier(**data)
            self.state.message_type = parsed.message_type
        except (json.JSONDecodeError, ValidationError, TypeError):
            # fallback: default to "logical"
            self.state.message_type = "logical"

        return self.state.message_type # which could be "emotional", "logical", or "exit"


    @router(classify_message)
    def route_message(self, message_type): # message_type is just a variable name for the return value of classify_message function passed by @router
        if message_type == "exit":
            return "exit"
        elif message_type == "emotional":
            return "therapist"
        else:
            return "logical"

    @listen("therapist")
    def therapist_response(self):
        messages = [
            {
                "role": "system",
                "content": """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                Show empathy, validate their feelings, and help them process their emotions.
                Ask thoughtful questions to help them explore their feelings more deeply.
                Avoid giving logical solutions unless explicitly asked."""
            }
        ]

        # Using .extend to add a list of dict (role, content) from conversation history to current message
        messages.extend(self.state.conversation_history)

        # Using .append to add a dict (role, content) from user message to current message
        messages.append({
            "role": "user",
            "content": self.state.user_message
        })
        
        reply = gemini_llm.call(messages)

        # Update conversation history
        self.state.conversation_history.extend([
            {"role": "user", "content": self.state.user_message},
            {"role": "assistant", "content": reply}
        ])

        self.state.response = reply
        print(f"Assistant: {reply}")
        return reply

    @listen("logical")
    def logical_response(self):
        messages = [
            {
                "role": "system",
                "content": """You are a purely logical assistant. Focus only on facts and information.
                Provide clear, concise answers based on logic and evidence.
                Do not address emotions or provide emotional support.
                Be direct and straightforward in your responses."""
            }
        ]
        
        # Add conversation history
        messages.extend(self.state.conversation_history)

        # Add current user message
        messages.append({
            "role": "user",
            "content": self.state.user_message
        })
        
        reply = gemini_llm.call(messages)

        # Update conversation history
        self.state.conversation_history.extend([
            {"role": "user", "content": self.state.user_message},
            {"role": "assistant", "content": reply}
        ])

        self.state.response = reply
        print(f"Assistant: {reply}")
        return reply

    @listen("exit")
    def handle_exit(self):
        print("Bye")
        return "goodbye"

def run_chatbot():
    flow = ChatbotFlow()
    while True:
        user_input = input("Type your message here or type exit to quit: ")
        if user_input.lower().strip() == "exit":
            break

        flow.state.user_message = user_input
        result = flow.kickoff()
        
        # Check if user wants to exit
        if flow.state.message_type == "exit" or result == "goodbye":
            break

if __name__ == "__main__":
    run_chatbot()