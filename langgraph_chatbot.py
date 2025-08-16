from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_message
from langchain.chat_models import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

# Initialize the Google Generative AI model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# Define the message structure by state
# Literal constrains a value to a fixed set of string(s)
class MessageClassifier(BaseModel):
    """A schema that return message_type and that it must be either "emotional" or "logical"."""
    message_type: Literal["emotional", "logical"] = Field(
        ..., 
        description="Classify if the message requires an emotional (therapist) or logical response."
    )

# Defines the shape of the graph State as a dictionary type.
class State(TypedDict):
    # Annotated attaches metadata (used later with add_messages)
    # add_messages: a reducer/merge function LangGraph provides for handling how the messages field should be combined across node outputs 
    # Annotated[list, add_messages] tells StateGraph how to combine multiple partial results for messages 
    # (e.g., append assistant reply to the message list).
    messages: Annotated[list, add_message] 
    message_type: str | None

# Define functions of each node
def classify_message(state: State) -> State:
    """Look at the last user message and classify it as "emotional" or "logical"."""
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier) # wraps the LLM so its output is parsed into the MessageClassifier Pydantic model.

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as either:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
            - 'logical': if it asks for facts, information, logical analysis, or practical solutions
            """
        },
        {"role": "user", "content": last_message}
    ])
    return {"message_type": result.message_type} # returns a partial state update. LangGraph will merge this into the running state.

def router(state: State) -> State:
    """Read the message_type and return a State (graph's conditional edges) use to route flow to the appropriate agent node."""
    message_type = state.get("message_type", "logical") # Default to logical if not be able to classify
    if message_type == "emotional":
        return {"next": "therapist"}
    else:
        return {"next": "logical"}

def therapist_agent(state: State) -> State:
    """Get the last message (from user), pass to therapist agent and return the assistant response"""
    last_message = state["messages"][-1]
    messages = [
        {"role": "system",
         "content": """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                        Show empathy, validate their feelings, and help them process their emotions.
                        Ask thoughtful questions to help them explore their feelings more deeply.
                        Avoid giving logical solutions unless explicitly asked."""},
         {"role": "user",
          "content": last_message.content}
    ]
    reply = llm.invoke(messages)

    # Returns a partial state containing one assistant message — the add_messages reducer will append this to the conversation.
    return {"messages": [{"role": "assistant", "content": reply.content}]}

def logical_agent(state: State) -> State:
    """Get the last message (from user), pass to logical agent and return the assistant response"""
    last_message = state["messages"][-1]
    messages = [
        {"role": "system",
         "content": """You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concise answers based on logic and evidence.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses."""},
         {"role": "user",
          "content": last_message.content}
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

# Define the graph builder
graph_builder = StateGraph(State)

# Define all the nodes in the graph
# A Node is a step in your workflow to performs an action or function using the current state.
# First parameter is node name, second is function that will run when the node is executed
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

# Define all the edges in the graph
# It is the connection between nodes, defines the flow of execution from one node to another.
graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {"therapist": "therapist", "logical": "logical"}
)

# Wrap it up and complie the graph
graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)
graph = graph_builder.compile()

# Chatbot function
def run_chatbot():
    """Chatbot to get user query and return AI response"""
    state = {"messages": [], "message_type": None} # Initialize state with an empty conversation.

    while True:
        user_input = input("Enter your message: ")
        if user_input.strip().lower() == "exit":
            print("Bye")
            break

        # Append the new user message to the messages list.
        # It is necessary when you are building the initial state before it enters the graph.
        # After that, add_messages will automatically merge them together inside the graph
        state["messages"] = state.get("messages", []) + [{"role": "user", "content": user_input}] 

        # Runs the graph (classifier → router → chosen agent), and LangGraph merges the returned partial state updates into a new current state.
        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")

if __name__ == "__main__":
    run_chatbot()