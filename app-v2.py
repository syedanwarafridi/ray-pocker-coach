from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
import streamlit as st

# Initialize Streamlit chat history
if "chatmem" not in st.session_state:
    st.session_state.chatmem = StreamlitChatMessageHistory(key="poker_coach_history")

###############################################################################
# 1. Shared State definition
###############################################################################

class ScenarioState(BaseModel):
    user_input: str
    scenario: Optional[str] = None  # JSON string from parser
    validated: bool = False
    clarification_needed: bool = False  # always False in this version
    solver_result: Optional[str] = None
    response: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None  # Added for history

###############################################################################
# 2. LLM factory (env var > fallback hard-coded)
###############################################################################


# def llm(temp: float = 0.0):
#     return ChatOpenAI(model="gpt-4o", temperature=temp, api_key=API_KEY)

key = ""
def llm(temp: float = 0.0):
    return ChatXAI(
        model="grok-4",
        xai_api_key=os.getenv("XAI_API_KEY"),
        temperature=temp,
        api_key=key,
        # search_parameters=dict(mode="auto", max_search_results=1)
    )

###############################################################################
# 3. Nodes
###############################################################################

# --- Parse ---------------------------------------------------------------
parse_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict JSON generator and a poker expert named Ray. Extract a poker hand scenario from the user text and output ONLY valid JSON.\n"
        "Include keys: players, action_history, hero_cards, board_cards, stack_sizes.\n"
        "If user describes an opponent’s tendencies (e.g., 'aggressive', 'tight'), include a field 'opponent_profile' with that info. Otherwise, omit it.\n"
        "Use the conversation history to inform context if relevant, but focus on the current input for the scenario.\n"
        "Conversation history: {chat_history}"
    ),
    ("human", "{input}"),
])

parser_llm = llm(0.15)

def node_parse(state: ScenarioState) -> Dict[str, Any]:
    history = state.chat_history or []
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    parsed = parser_llm.invoke(parse_prompt.format_messages(input=state.user_input, chat_history=history_str)).content
    return {"scenario": parsed, "validated": False}

# --- Validate (always OK / no questions) ---------------------------------
validate_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You're Ray, a sharp poker coach with a friendly vibe. Take a look at the scenario and give it a quick thumbs up if it's plausible. "
        "Assume reasonable defaults where needed. Don't ask any follow-ups—just confirm it's good to go with a simple 'OK'."
    ),
    ("human", "{scenario}"),
])

validator_llm = llm(0.0)

def node_validate(state: ScenarioState) -> Dict[str, Any]:
    _ = validator_llm.invoke(validate_prompt.format_messages(scenario=state.scenario)).content
    return {"validated": True, "clarification_needed": False}

# --- Solve ---------------------------------------------------------------
solve_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You're Ray, the cool poker-savvy llama who makes game theory simple. Based on the JSON hand scenario and player input, do the following:\n"
        "• Estimate pot size and effective stack.\n"
        "• Compute SPR (stack-to-pot ratio).\n"
        "• List ONLY the hero’s legal actions right now.\n"
        "• Recommend action frequencies for legal actions. Keep it digestible — round percentages, and skip %s if they clutter the message.\n"
        "• If user mentions a read (e.g., villain is aggressive), make a clear exploit adjustment and explain it briefly.\n"
        "• Include EVs for the top 2 actions if useful.\n"
        "Be confident but not robotic — you’re a helpful poker buddy, not a professor."
    ),
    ("human", "{scenario}"),
])

solver_llm = llm(0.1)

def node_solve(state: ScenarioState) -> Dict[str, Any]:
    result = solver_llm.invoke(solve_prompt.format_messages(scenario=state.scenario)).content
    return {"solver_result": result}

# --- Respond -------------------------------------------------------------
respond_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You're Ray, the confident but chill poker buddy who simplifies tough spots. Summarize the solver_result clearly in 1–2 casual, punchy sentences.\n"
        "• Use natural speech — this is bar chat, not a lecture.\n"
        "• Mention action %s only if they add clarity.\n"
        "• Recommend only actions the hero can actually take.\n"
        "• If opponent tendencies were given, adjust the advice and say why in simple terms.\n"
        "• Throw in one of your signature phrases like: 'Honestly', 'Solid play', 'Here’s the deal', 'Easy bet here', or 'Prints money' — if it fits naturally.\n"
        "• If relevant, reference the conversation history to make the response feel contextual and personalized.\n"
        "Conversation history: {chat_history}"
    ),
    ("human", "{solver_result}"),
])

responder_llm = llm(0.4)

def node_respond(state: ScenarioState) -> Dict[str, Any]:
    history = state.chat_history or []
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    reply = responder_llm.invoke(respond_prompt.format_messages(solver_result=state.solver_result, chat_history=history_str)).content
    return {"response": reply}

###############################################################################
# 4. Build graph
###############################################################################

def build_graph() -> StateGraph:
    g = StateGraph(ScenarioState)
    g.add_node("parse", node_parse)
    g.add_node("validate", node_validate)
    g.add_node("solve", node_solve)
    g.add_node("respond", node_respond)

    g.set_entry_point("parse")
    g.add_edge("parse", "validate")
    g.add_edge("validate", "solve")
    g.add_edge("solve", "respond")
    g.add_edge("respond", END)
    return g.compile()

graph = build_graph()

###############################################################################
# 5. Helper
###############################################################################

def run_turn(msg: str, session_id: str = "default") -> str:
    # Retrieve chat history
    chat_history = [
        {"role": "user" if msg.type == "human" else "assistant", "content": msg.content}
        for msg in st.session_state.chatmem.messages
    ]
    
    # Run graph with history
    result = graph.invoke({"user_input": msg, "chat_history": chat_history})
    response = result.get("response", "(no response)")
    
    # Append to Streamlit chat history
    st.session_state.chatmem.add_user_message(msg)
    st.session_state.chatmem.add_ai_message(response)
    return response

###############################################################################
# 6. Streamlit UI (stateless)
###############################################################################

def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="Poker Coach", page_icon="🃏", layout="wide")
    st.title("🃏 Multi-Agent Poker Coach")

    # Sidebar for API key override
    with st.sidebar:
        st.header("🔑 API Key")
        key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        if key:
            os.environ["OPENAI_API_KEY"] = key
            st.success("Key set for this session.")

    # Display chat history
    for msg in st.session_state.chatmem.messages:
        role = "assistant" if msg.type == "ai" else "user"
        with st.chat_message(role):
            st.markdown(msg.content)

    prompt = st.chat_input("Describe a poker spot …")
    if prompt:
        run_turn(prompt, session_id="default")
        st.rerun()

###############################################################################
# 7. Entrypoint
###############################################################################

if __name__ == "__main__":
    run_streamlit()