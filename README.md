# CrewAI Flow vs LangGraph: A Comparative Guide

## Introduction
This repository contains two implementations of a simple chatbot that classifies user input as either **emotional** or **logical**, then routes the request to the appropriate response generator. The example in here is inspired by a [Langgraph tutorial video](https://youtu.be/1w5cCXlh7JQ?si=dpgxggk4LBXP3XCi) by Tech With Tim on his Youtube channel

- **[CrewAI Flow](https://docs.crewai.com/en/concepts/flows)** is an orchestration framework for AI workflows, using decorators like `@start`, `@router`, and `@listen` to define flow logic.
- **[LangGraph](https://python.langchain.com/docs/langgraph)** is a stateful graph-based orchestration library, extending LangChain to manage AI agent workflows using nodes and edges.

Both aim to streamline AI application design but differ in **architecture**, **implementation style**, and **workflow modeling**.

---

## Similarities

| Feature | CrewAI Flow | LangGraph |
|---------|-------------|-----------|
| **Structured State** | Uses Pydantic `BaseModel` (`MessageState`) to define state fields | Uses `TypedDict` + `Annotated` for state schema |
| **LLM Integration** | Supports direct model calls with structured output via `response_format` | Uses `.with_structured_output()` for parsing into Pydantic models |
| **Routing** | Both route messages to different response handlers based on classification | Same routing goal |
| **Conversation Handling** | Both maintain conversation history across turns | Implemented differently (manual vs reducer) |
| **Multi-Agent Flow** | Support multiple “agent” nodes/functions | Similar agent separation |

---

## Differences

### 1. **Architecture**
- **CrewAI Flow**
  - Workflow defined via **decorators** (`@start`, `@router`, `@listen`).
  - Execution proceeds through **function chaining** inside a single Flow class.
  - State is strongly typed via a Pydantic model, or unstructured and can be added or updated during the execution.
  - You can have multiple `@start()` methods in a Flow, and they will all be executed when the Flow is started.
- **LangGraph**
  - Workflow is modeled as a **graph** with **nodes** (functions) and **edges** (execution paths).
  - Uses `StateGraph` to define the topology and `graph.invoke()` to execute.
  - State merges via reducers like `add_message`.

### 2. **State Management**
- **CrewAI Flow**: Manual history management using `.append()` and `.extend()` inside handlers. Or using `@persist` decorator enables automatic state persistence in CrewAI Flows, allowing you to maintain flow state across restarts or different workflow executions.
- **LangGraph**: Automatic message accumulation using `add_message` reducer.

### 3. **Routing**
- **CrewAI Flow**: Routes directly via return values from the `@router` function.
- **LangGraph**: Uses `add_conditional_edges()` for conditional branching.

### 4. **Conditional Logic**
- **CrewAI Flow**: Using decorators such as `@and_` & `@or_` to to listen to multiple methods and trigger the listener method
- **LangGraph**: to be added later

### 5. **Conversation History**

#### CrewAI Flow (manual)
```python
messages.extend(self.state.conversation_history)
messages.append({"role": "user", "content": self.state.user_message})
self.state.conversation_history.extend([
    {"role": "user", "content": self.state.user_message},
    {"role": "assistant", "content": reply}
])
```
#### CrewAI Flow (automatic and retain across sessions)
```python
@persist  # Using SQLiteFlowPersistence by default
class MyFlow(Flow[MyState]):
    @start()
    def initialize_flow(self):
        # This method will automatically have its state persisted
        self.state.counter = 1
        print("Initialized flow. State ID:", self.state.id)

    @listen(initialize_flow)
    def next_step(self):
        # The state (including self.state.id) is automatically reloaded
        self.state.counter += 1
        print("Flow state is persisted. Counter:", self.state.counter)
```

#### LangGraph (automatic)
```python
state["messages"] = state.get("messages", []) + [{"role": "user", "content": user_input}]
# add_message reducer auto-appends assistant replies inside the graph
```

---

## Use Cases

### CrewAI Flow Best For:
- Developers who prefer **Pythonic OOP with decorators**.
- Flows where **manual control** over state is needed.
- Complex session handling, especially when persisting state across runs with `@persist`.
- Rapid prototyping where code reads top-to-bottom like a story.

### LangGraph Best For:
- **Complex workflows** requiring **dynamic branching** and graph topology.
- Scenarios with **multiple concurrent or parallel agents**.
- Applications where **state merging** should be automated.
- Visualizing and debugging workflows as graphs.

---

## Conclusion
Both **CrewAI Flow** and **LangGraph** offer robust ways to orchestrate AI agent workflows:

- **Choose CrewAI Flow** if you want a **decorator-based, class-oriented** approach with fine-grained control over conversation state.
- **Choose LangGraph** if you need **graph-based orchestration** with built-in state merging and flexible branching.

| Criterion          | CrewAI Flow | LangGraph |
|--------------------|-------------|-----------|
| **Learning Curve** | Easier for Python OOP devs | Easier for graph/flowchart thinkers |
| **State Control**  | Manual, flexible | Automatic, consistent |
| **Workflow Model** | Sequential functions | Graph-based nodes |
| **Best For**       | Controlled, simple-to-mid complexity | Complex, branching multi-agent workflows |

Both frameworks can achieve similar results — your choice should depend on **workflow complexity**, **team familiarity**, and **state management preferences**.
