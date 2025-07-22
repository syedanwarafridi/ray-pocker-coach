"""multi_agent_workflow.py  + simple Streamlit front‑end
-----------------------------------------------------------------
A self‑contained, stateless multi‑agent poker assistant that:
1. Parses a natural‑language spot into JSON
2. **Never** asks follow‑up questions – it fills reasonable defaults
3. Runs a "solver" agent that returns betting‑frequency math (EV, % splits)
4. Responds in ≤ 3 friendly sentences, including the key numbers

Launch UI:
    streamlit run multi_agent_workflow.py
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


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

###############################################################################
# 2. LLM factory (env var > fallback hard‑coded)
###############################################################################

API_KEY = ""

def llm(temp: float = 0.0):
    return ChatOpenAI(model="gpt-4o-mini", temperature=temp, api_key=API_KEY)

###############################################################################
# 3. Nodes
###############################################################################

# --- Parse ---------------------------------------------------------------
parse_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict JSON generator. Extract a poker hand scenario from the user text and output ONLY valid JSON.\n"
        "Include keys: players, action_history, hero_cards, board_cards, stack_sizes.\n"
        "If user describes an opponent’s tendencies (e.g., 'aggressive', 'tight'), include a field 'opponent_profile' with that info. Otherwise, omit it."
    ),
    ("human", "{input}"),
])


parser_llm = llm(0.15)

def node_parse(state: ScenarioState) -> Dict[str, Any]:
    parsed = parser_llm.invoke(parse_prompt.format_messages(input=state.user_input)).content
    return {"scenario": parsed, "validated": False}

# --- Validate (always OK / no questions) ---------------------------------
# validate_prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a poker expert. If the provided scenario is plausible, reply only 'OK'. Assume reasonable defaults for any missing info. NEVER ask questions."),
#     ("human", "{scenario}"),
# ])
validate_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You're Ray, a sharp poker coach with a friendly vibe. Take a look at the scenario and give it a quick thumbs up if it's plausible. "
     "Assume reasonable defaults where needed. Don't ask any follow-ups—just confirm it's good to go with a simple 'OK'.")
])

validator_llm = llm(0.0)

def node_validate(state: ScenarioState) -> Dict[str, Any]:
    _ = validator_llm.invoke(validate_prompt.format_messages(scenario=state.scenario)).content
    # Always proceed
    return {"validated": True, "clarification_needed": False}

# --- Solve ----------------------------------------------------------------
# solve_prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         "You are a professional poker solver. Given a JSON scenario and natural-language user input, do the following:\n"
#         "• Estimate pot size and effective stack.\n"
#         "• Compute stack‑to‑pot ratio (SPR).\n"
#         "• Determine ONLY the actions currently available to the hero based on the action history (e.g., if villain checked, hero can bet or check, but not call).\n"
#         "• Provide recommended action frequencies (%, rounded to one decimal) ONLY for legal actions, **unless the user input implies a strong exploitative tendency (e.g., 'opponent is very aggressive')**.\n"
#         "• If exploitative info is provided, adjust from equilibrium accordingly and explain the shift briefly.\n"
#         "• Show EV estimates for the top 2 legal actions if helpful.\n"
#         "• Avoid including percentages if they're not useful for the situation — use judgment based on clarity."
#     ),
#     ("human", "{scenario}"),
# ])
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

# --- Respond --------------------------------------------------------------
respond_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You're Ray, the confident but chill poker buddy who simplifies tough spots. Summarize the solver_result clearly in 2–3 casual, punchy sentences.\n"
        "• Use natural speech — this is bar chat, not a lecture.\n"
        "• Mention action %s only if they add clarity.\n"
        "• Recommend only actions the hero can actually take.\n"
        "• If opponent tendencies were given, adjust the advice and say why in simple terms.\n"
        "• Throw in one of your signature phrases like: 'Honestly', 'Solid play', 'Here’s the deal', 'Easy bet here', or 'Prints money' — if it fits naturally.\n"
        "Stay confident, stay helpful, and always sound like a skilled friend, not a condescending expert."
    ),
    ("human", "{solver_result}"),
])



responder_llm = llm(0.4)

def node_respond(state: ScenarioState) -> Dict[str, Any]:
    reply = responder_llm.invoke(respond_prompt.format_messages(solver_result=state.solver_result)).content
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
    g.add_edge("validate", "solve")  # no clarifications branch
    g.add_edge("solve", "respond")
    g.add_edge("respond", END)
    return g.compile()

graph = build_graph()

###############################################################################
# 5. Helper
###############################################################################

def run_turn(msg: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
    result = graph.invoke({"user_input": msg})
    if history is not None:
        history.append(("user", msg))
        history.append(("assistant", result.get("response", "")))
    return result.get("response", "(no response)")

###############################################################################
# 6. Streamlit UI (stateless)
###############################################################################

def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="Poker Coach", page_icon="🃏", layout="wide")
    st.title("🃏 Multi‑Agent Poker Coach")

    # Sidebar for API key override
    with st.sidebar:
        st.header("🔑 API Key")
        key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        if key:
            os.environ["OPENAI_API_KEY"] = key
            st.success("Key set for this session.")

    if "chat" not in st.session_state:
        st.session_state.chat = []  # type: List[Tuple[str, str]]

    # Display history
    for role, text in st.session_state.chat:
        with st.chat_message("assistant" if role == "assistant" else "user"):
            st.markdown(text)

    prompt = st.chat_input("Describe a poker spot …")
    if prompt:
        run_turn(prompt, history=st.session_state.chat)
        st.rerun()

###############################################################################
# 7. Entrypoint
###############################################################################

if __name__ == "__main__":
    run_streamlit()