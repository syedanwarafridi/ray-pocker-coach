from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from langchain_xai import ChatXAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langgraph.checkpoint.memory import MemorySaver
# from retriver import Retriever
from langchain_core.tools import tool
import streamlit as st

# Initialize Streamlit chat history
if "chatmem" not in st.session_state:
    st.session_state.chatmem = StreamlitChatMessageHistory(key="poker_coach_history")

###############################################################################
# 1. Shared State definition
###############################################################################

class ScenarioState(BaseModel):
    user_input: str
    scenario: Optional[str] = None  # Directly use user input as scenario
    solver_result: Optional[str] = None
    response: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None


###############################################################################
# 3. Tools
###############################################################################
   
# @tool
# def top_docs_retriever(query: str) -> str:
#     """Returns top 3 relevant document chunks from the poker knowledgebase."""
#     r = Retriever()
#     docs = r.query(query, top_k=1)
#     return "\n\n".join([f"{d['text']}" for d in docs])

###############################################################################
# 2. LLM factory using Grok-4
###############################################################################

# Use hardcoded or environment API key
key = ""

# def llm_with_tool(temp: float = 0.0):
#     llm = ChatXAI(
#         model="grok-4",
#         xai_api_key=os.getenv("XAI_API_KEY"),
#         temperature=temp,
#         api_key=key,
#     )
#     return llm.bind_tools([top_docs_retriever])

def llm(temp: float = 0.0):
    return ChatXAI(
        model="grok-4",
        xai_api_key=os.getenv("XAI_API_KEY"),
        temperature=temp,
        api_key=key,
    )

###############################################################################
# 4. Nodes
###############################################################################

# --- Solve Node ---
solve_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You're **Ray**, a poker-savvy llama with elite logic and killer instincts. The user has provided a specific poker scenario:\n\n"
     "**Scenario:** {scenario}\n\n"
     "Break down the hand with precision, step by step:\n\n"
     "**1. Pot Size & Effective Stacks:**\n"
     "- Calculate the **current pot size** and the **effective stack**.\n"
     "- Be precise with bet sizes and positions.\n\n"
     "**2. SPR (Stack-to-Pot Ratio):**\n"
     "- Compute SPR = effective stack / pot size.\n"
     "- Briefly explain what a high or low SPR means for playability.\n\n"
     "**3. Legal Actions:**\n"
     "- List all legal options for Hero: fold, call, raise (with sizing recommendations).\n\n"
     "**4. EV & Frequencies:**\n"
     "- Estimate the **Expected Value (EV)** in chips for each action.\n"
     "- Recommend action **frequencies** (e.g., 'call 70%, raise 30%'), explaining **why**.\n\n"
     "**5. Reads & Adjustments:**\n"
     "- If villain type is known (e.g., tight, loose, aggressive), adjust the recommended strategy accordingly.\n\n"

     "Deliver a clear, logical analysis based solely on the provided scenario. Close with a bullet-point summary of recommended action(s) and frequencies."
     "Always directly address every part of the user's question, especially when multiple hands, actions, or comparisons are involved."
    )
])

solver_llm = llm(0.1)

def node_solve(state: ScenarioState) -> Dict[str, Any]:
    result = solver_llm.invoke(solve_prompt.format_messages(scenario=state.user_input)).content
    return {"solver_result": result}

# --- Respond Node ---
respond_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You're **Ray**—the poker buddy who keeps it real. You take even complex solver results and explain them like it's no big deal.\n\n"
     "Here’s your job:\n"
     "• Summarize the solver_result in **1–2 punchy sentences**.\n"
     "• Speak like a confident, chill friend who knows their poker.\n\n"
     "**Your tone:**\n"
     "- 🃏 Confident but approachable\n"
     "- ✂️ Concise and action-driven\n"
     "- 🎯 No fluff, just clarity\n"
     "- 💬 Conversational and fun, but not goofy\n"
     "- 🎓 Lightly educational if it helps understanding\n\n"
     "**Rules for your response:**\n"
     "• No 'solver says' or robotic phrasing\n"
     "• Use casual openers like: 'Here’s the deal', 'Easy raise here', 'Honestly', 'Snap call that', 'Trust the process'\n"
     "• Mention action frequencies only if they help clarity (e.g., 'mostly call, mix in raises')\n"
     "• If reads were considered, weave that in naturally (e.g., 'Against nits, this is pure fold')\n"
     "• Give a nudge of motivation if the spot feels tricky\n\n"
     "**Goal:** Keep poker fun, sharp, and empowering. Your summary should make even a confusing spot feel like a smart, confident decision."
    ),
    ("human", "{solver_result}")
])

# respond_prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         "You're Ray, the confident but chill poker buddy who simplifies tough spots. Summarize the solver_result clearly in 1–2 casual, punchy sentences.\n"
#         "• Use natural speech — this is bar chat, not a lecture.\n"
#         "• Mention action %s only if they add clarity.\n"
#         "• Recommend only actions the hero can actually take.\n"
#         "• If opponent tendencies were given, adjust the advice and say why in simple terms.\n"
#         "• Throw in one of your signature phrases like: 'Honestly', 'Solid play', 'Here’s the deal', 'Easy bet here', or 'Prints money' — if it fits naturally.\n"
#         "• If relevant, reference the conversation history to make the response feel contextual and personalized.\n"
#         "Conversation history: {chat_history}"
#     ),
#     ("human", "{solver_result}"),
# ])
responder_llm = llm(0.5)

def node_respond(state: ScenarioState) -> Dict[str, Any]:
    history = state.chat_history or []
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    reply = responder_llm.invoke(
        respond_prompt.format_messages(
            solver_result=state.solver_result,
            chat_history=history_str
        )
    ).content
    return {"response": reply}

###############################################################################
# 4. Build simplified graph
###############################################################################

def build_graph() -> StateGraph:
    g = StateGraph(ScenarioState)
    g.add_node("solve", node_solve)
    g.add_node("respond", node_respond)
    g.set_entry_point("solve")
    g.add_edge("solve", "respond")
    g.add_edge("respond", END)
    return g.compile()

graph = build_graph()

###############################################################################
# 5. Helper
###############################################################################

def run_turn(msg: str, session_id: str = "default") -> str:
    chat_history = [
        {"role": "user" if msg.type == "human" else "assistant", "content": msg.content}
        for msg in st.session_state.chatmem.messages
    ]

    result = graph.invoke({"user_input": msg, "chat_history": chat_history})
    response = result.get("response", "(no response)")

    st.session_state.chatmem.add_user_message(msg)
    st.session_state.chatmem.add_ai_message(response)
    return response

###############################################################################
# 6. Streamlit UI
###############################################################################

def run_streamlit():
    st.set_page_config(page_title="Poker Coach", page_icon="🃏", layout="wide")
    st.title("🃏 Multi-Agent Poker Coach")

    with st.sidebar:
        st.header("🔑 API Key")
        key_input = st.text_input("XAI API Key", type="password", value=os.getenv("XAI_API_KEY", ""))
        if key_input:
            os.environ["XAI_API_KEY"] = key_input
            st.success("Key set for this session.")

    for msg in st.session_state.chatmem.messages:
        role = "assistant" if msg.type == "ai" else "user"
        with st.chat_message(role):
            st.markdown(msg.content)

    prompt = st.chat_input("Describe a poker spot …")
    if prompt:
        run_turn(prompt, session_id="default")
        st.rerun()

if __name__ == "__main__":
    run_streamlit()

# from __future__ import annotations

# import os
# import logging
# from typing import Any, Dict, List, Optional, TypedDict
# from pydantic import BaseModel
# from langchain_xai import ChatXAI
# from langchain_core.prompts import ChatPromptTemplate
# from langgraph.graph import StateGraph, END
# from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
# from langgraph.checkpoint.memory import MemorySaver
# from retriver import Retriever
# from langchain_core.tools import tool
# import streamlit as st

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler()]
# )
# logger = logging.getLogger(__name__)

# # Initialize Streamlit chat history
# if "chatmem" not in st.session_state:
#     st.session_state.chatmem = StreamlitChatMessageHistory(key="poker_coach_history")

# ###############################################################################
# # 1. Shared State definition
# ###############################################################################

# class ScenarioState(TypedDict):
#     user_input: str
#     scenario: Optional[str]  # Directly use user input as scenario
#     solver_result: Optional[str]
#     response: Optional[str]
#     chat_history: Optional[List[Dict[str, str]]]
#     tool_result: Optional[str]  # Store tool call results

# ###############################################################################
# # 2. LLM factory using Grok-4
# ###############################################################################


# def llm_with_tool(temp: float = 0.0):
#     logger.info("Initializing LLM with tool binding")
#     llm = ChatXAI(
#         model="grok-4",
#         xai_api_key=os.getenv("XAI_API_KEY"),
#         temperature=temp,
#         api_key=key,
#         # parallel_tool_calls=False,
#     )
#     return llm.bind_tools([top_docs_retriever])

# def llm(temp: float = 0.0):
#     logger.info("Initializing LLM without tool binding")
#     return ChatXAI(
#         model="grok-4",
#         xai_api_key=os.getenv("XAI_API_KEY"),
#         temperature=temp,
#         api_key=key,
#     )

# ###############################################################################
# # 3. Tools
# ###############################################################################

# @tool(return_direct=True)
# def top_docs_retriever(query: str) -> str:
#     """Returns top 3 relevant document chunks from the poker knowledgebase."""
#     logger.info(f"Executing top_docs_retriever with query: {query}")
#     r = Retriever()
#     docs = r.query(query, top_k=1)
#     result = "\n\n".join([f"{d['text']}" for d in docs])
#     logger.info(f"Tool result: {result}")
#     return result

# ###############################################################################
# # 4. Nodes
# ###############################################################################

# # --- Solve Node ---
# solve_prompt = ChatPromptTemplate.from_messages([
#     ("system",
#      "You're **Ray**, a poker-savvy llama with elite logic and killer instincts. The user has provided a specific poker scenario:\n\n"
#      "**Scenario:** {scenario}\n\n"
#      "Break down the hand with precision, step by step:\n\n"
#      "**1. Pot Size & Effective Stacks:**\n"
#      "- Calculate the **current pot size** and the **effective stack**.\n"
#      "- Be precise with bet sizes and positions.\n\n"
#      "**2. SPR (Stack-to-Pot Ratio):**\n"
#      "- Compute SPR = effective stack / pot size.\n"
#      "- Briefly explain what a high or low SPR means for playability.\n\n"
#      "**3. Legal Actions:**\n"
#      "- List all legal options for Hero: fold, call, raise (with sizing recommendations).\n\n"
#      "**4. EV & Frequencies:**\n"
#      "- Estimate the **Expected Value (EV)** in chips for each action.\n"
#      "- Recommend action **frequencies** (e.g., 'call 70%, raise 30%'), explaining **why**.\n\n"
#      "**5. Reads & Adjustments:**\n"
#      "- If villain type is known (e.g., tight, loose, aggressive), adjust the recommended strategy accordingly.\n\n"
#      "**6. Tool Usage (if relevant, Only once, do not call tool again and again):**\n"
#      "- Use the `top_docs_retriever` tool **only** if specific poker data (like preflop ranges, solver-approved lines, or board texture insights) is needed.\n"
#      "- If the tool result is unhelpful, ignore it and rely on your strategic knowledge.\n\n"
#      "**Tool Result (if available):**\n{tool_result}\n\n"
#      "Deliver a clear, logical analysis based solely on the provided scenario. Close with a bullet-point summary of recommended action(s) and frequencies."
#      "Always directly address every part of the user's question, especially when multiple hands, actions, or comparisons are involved."
#     )
# ])


# solver_llm = llm_with_tool(0.1)

# def node_solve(state: ScenarioState) -> Dict[str, Any]:
#     logger.info(f"Solve node started with user_input: {state['user_input']}")
#     messages = solve_prompt.format_messages(
#         scenario=state["user_input"],
#         tool_result=state.get("tool_result", "No tool result available")
#     )
#     logger.info("Invoking solver LLM")
#     result = solver_llm.invoke(messages)
    
#     # Log the raw LLM response for debugging
#     logger.info(f"Raw solver LLM response: {result}")
    
#     # Check if the LLM requested a tool call
#     if hasattr(result, "tool_calls") and result.tool_calls:
#         logger.info(f"Solver requested tool call: {result.tool_calls}")
#         return {"tool_result": None}  # Trigger tool node
#     logger.info(f"Solver result: {result.content}")
#     return {"solver_result": result.content}

# # --- Tool Node ---
# def node_tool(state: ScenarioState) -> Dict[str, Any]:
#     logger.info("Tool node started")
#     solver_response = solver_llm.invoke(
#         solve_prompt.format_messages(
#             scenario=state["user_input"],
#             tool_result=state.get("tool_result", "No tool result available")
#         )
#     )
#     logger.info(f"Tool node solver response: {solver_response}")
#     if hasattr(solver_response, "tool_calls") and solver_response.tool_calls:
#         tool_call = solver_response.tool_calls[0]
#         if tool_call["name"] == "top_docs_retriever":
#             query = tool_call["args"]["query"]
#             logger.info(f"Tool node invoking top_docs_retriever with query: {query}")
#             tool_result = top_docs_retriever.invoke({"query": query})
#             logger.info(f"Tool node result: {tool_result}")
#             return {"tool_result": tool_result}
#     logger.info("No relevant tool call found")
#     return {"tool_result": "No relevant tool result"}

# # --- Respond Node ---
# respond_prompt = ChatPromptTemplate.from_messages([
#     ("system",
#      "You're **Ray**—the poker buddy who keeps it real. You take even complex solver results and explain them like it's no big deal.\n\n"
#      "Here’s your job:\n"
#      "• Summarize the solver_result in **1–2 punchy sentences**.\n"
#      "• Speak like a confident, chill friend who knows their poker.\n\n"
#      "**Your tone:**\n"
#      "- 🃏 Confident but approachable\n"
#      "- ✂️ Concise and action-driven\n"
#      "- 🎯 No fluff, just clarity\n"
#      "- 💬 Conversational and fun, but not goofy\n"
#      "- 🎓 Lightly educational if it helps understanding\n\n"
#      "**Rules for your response:**\n"
#      "• No 'solver says' or robotic phrasing\n"
#      "• Use casual openers like: 'Here’s the deal', 'Easy raise here', 'Honestly', 'Snap call that', 'Trust the process'\n"
#      "• Mention action frequencies only if they help clarity (e.g., 'mostly call, mix in raises')\n"
#      "• If reads were considered, weave that in naturally (e.g., 'Against nits, this is pure fold')\n"
#      "• Give a nudge of motivation if the spot feels tricky\n\n"
#      "**Goal:** Keep poker fun, sharp, and empowering. Your summary should make even a confusing spot feel like a smart, confident decision."
#     ),
#     ("human", "{solver_result}")
# ])


# responder_llm = llm(0.5)

# def node_respond(state: ScenarioState) -> Dict[str, Any]:
#     logger.info("Respond node started")
#     history = state["chat_history"] or []
#     history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
#     logger.info(f"Chat history: {history_str}")
#     reply = responder_llm.invoke(
#         respond_prompt.format_messages(
#             solver_result=state["solver_result"],
#             chat_history=history_str
#         )
#     ).content
#     logger.info(f"Respond node result: {reply}")
#     return {"response": reply}

# ###############################################################################
# # 5. Build graph with tool node
# ###############################################################################

# def build_graph() -> StateGraph:
#     logger.info("Building graph")
#     g = StateGraph(ScenarioState)
#     g.add_node("solve", node_solve)
#     g.add_node("tool", node_tool)
#     g.add_node("respond", node_respond)
    
#     g.set_entry_point("solve")
    
#     # Conditional routing: if solver requests a tool call, go to tool node
#     def route_to_tool(state: ScenarioState) -> str:
#         if state.get("tool_result") is None and state.get("solver_result") is None:
#             logger.info("Routing to tool node")
#             return "tool"
#         logger.info("Routing to respond node")
#         return "respond"
    
#     g.add_conditional_edges("solve", route_to_tool, {"tool": "tool", "respond": "respond"})
#     g.add_edge("tool", "solve")  # Return to solver after tool execution
#     g.add_edge("respond", END)
    
#     logger.info("Graph compiled with MemorySaver")
#     return g.compile(checkpointer=MemorySaver())

# graph = build_graph()

# ###############################################################################
# # 6. Helper
# ###############################################################################

# def run_turn(msg: str, session_id: str = "default") -> str:
#     logger.info(f"Starting turn with user input: {msg}")
#     chat_history = [
#         {"role": "user" if msg.type == "human" else "assistant", "content": msg.content}
#         for msg in st.session_state.chatmem.messages
#     ]
#     logger.info(f"Chat history for turn: {chat_history}")

#     result = graph.invoke(
#         {"user_input": msg, "chat_history": chat_history},
#         config={"configurable": {"thread_id": session_id}}
#     )
#     response = result.get("response", "(no response)")
#     logger.info(f"Turn completed with response: {response}")

#     st.session_state.chatmem.add_user_message(msg)
#     st.session_state.chatmem.add_ai_message(response)
#     return response

# ###############################################################################
# # 7. Streamlit UI
# ###############################################################################

# def run_streamlit():
#     st.set_page_config(page_title="Poker Coach", page_icon="🃏", layout="wide")
#     st.title("🃏 Multi-Agent Poker Coach")

#     with st.sidebar:
#         st.header("🔑 API Key")
#         key_input = st.text_input("XAI API Key", type="password", value=os.getenv("XAI_API_KEY", ""))
#         if key_input:
#             os.environ["XAI_API_KEY"] = key_input
#             st.success("Key set for this session.")

#     for msg in st.session_state.chatmem.messages:
#         role = "assistant" if msg.type == "ai" else "user"
#         with st.chat_message(role):
#             st.markdown(msg.content)

#     prompt = st.chat_input("Describe a poker spot …")
#     if prompt:
#         logger.info("Received user prompt in Streamlit UI")
#         run_turn(prompt, session_id="default")
#         st.rerun()

# if __name__ == "__main__":
#     run_streamlit()