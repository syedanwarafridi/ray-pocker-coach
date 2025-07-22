from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import logging

from pydantic import BaseModel
from langchain_xai import ChatXAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langgraph.checkpoint.memory import MemorySaver
# from retriver import Retriever
from langchain_core.tools import tool
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Streamlit chat history
if "chatmem" not in st.session_state:
    logger.info("Initializing Streamlit chat history")
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
# 2. LLM factory using Grok-4
###############################################################################

# Use hardcoded or environment API key
key = ""


def llm(temp: float = 0.0):
    logger.info(f"Initializing LLM, temperature={temp}")
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
    logger.info(f"Executing solve node with user_input: {state.user_input}")
    result = solver_llm.invoke(solve_prompt.format_messages(scenario=state.user_input)).content
    logger.info("Solve node completed")
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

responder_llm = llm(0.5)

def node_respond(state: ScenarioState) -> Dict[str, Any]:
    logger.info("Executing respond node")
    history = state.chat_history or []
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    reply = responder_llm.invoke(
        respond_prompt.format_messages(
            solver_result=state.solver_result,
            chat_history=history_str
        )
    ).content
    logger.info("Respond node completed")
    return {"response": reply}

###############################################################################
# 4. Build simplified graph
###############################################################################

def build_graph() -> StateGraph:
    logger.info("Building StateGraph")
    g = StateGraph(ScenarioState)
    g.add_node("solve", node_solve)
    g.add_node("respond", node_respond)
    g.set_entry_point("solve")
    g.add_edge("solve", "respond")
    g.add_edge("respond", END)
    compiled_graph = g.compile()
    logger.info("StateGraph compiled")
    return compiled_graph

graph = build_graph()

###############################################################################
# 5. Helper
###############################################################################

def run_turn(msg: str, session_id: str = "default") -> str:
    logger.info(f"Running turn with message: {msg}")
    chat_history = [
        {"role": "user" if msg.type == "human" else "assistant", "content": msg.content}
        for msg in st.session_state.chatmem.messages
    ]

    result = graph.invoke({"user_input": msg, "chat_history": chat_history})
    response = result.get("response", "(no response)")
    logger.info(f"Graph invocation completed, response: {response}")

    st.session_state.chatmem.add_user_message(msg)
    st.session_state.chatmem.add_ai_message(response)
    logger.info("Chat history updated")
    return response

###############################################################################
# 6. Streamlit UI
###############################################################################

def run_streamlit():
    logger.info("Starting Streamlit UI")
    st.set_page_config(page_title="Poker Coach", page_icon="🃏", layout="wide")
    st.title("🃏 Multi-Agent Poker Coach")

    with st.sidebar:
        st.header("🔑 API Key")
        key_input = st.text_input("XAI API Key", type="password", value=os.getenv("XAI_API_KEY", ""))
        if key_input:
            os.environ["XAI_API_KEY"] = key_input
            logger.info("API key set in Streamlit session")
            st.success("Key set for this session.")

    for msg in st.session_state.chatmem.messages:
        role = "assistant" if msg.type == "ai" else "user"
        with st.chat_message(role):
            st.markdown(msg.content)

    prompt = st.chat_input("Describe a poker spot …")
    if prompt:
        logger.info(f"Received user prompt: {prompt}")
        run_turn(prompt, session_id="default")
        logger.info("Rerunning Streamlit app")
        st.rerun()

if __name__ == "__main__":
    logger.info("Launching Poker Coach application")
    run_streamlit()