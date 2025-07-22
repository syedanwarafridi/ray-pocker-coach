from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import logging
from langchain_xai import ChatXAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from database import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize memory and database
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
db_manager = DatabaseManager()

class ScenarioState(BaseModel):
    user_input: str
    scenario: Optional[str] = None
    solver_result: Optional[str] = None
    response: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None

def llm(temp: float = 0.0):
    key = ""
    logger.info(f"Initializing LLM, temperature={temp}")
    return ChatXAI(
        model="grok-4",
        xai_api_key=os.getenv("XAI_API_KEY"),
        temperature=temp,
        api_key=key,
    )

def convert_message_to_dict(message: Any) -> Dict[str, str]:
    """Convert a LangChain message (HumanMessage or AIMessage) to a dictionary."""
    return {
        "role": "human" if isinstance(message, HumanMessage) else "ai",
        "content": message.content
    }

def build_poker_solver_graph() -> Any:
    solve_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You're **Ray**, a poker-savvy llama with elite logic and killer instincts. The user has provided a specific poker scenario:\n\n"
         "**Scenario:** {scenario}\n\n"
         "**Conversation History:**\n{chat_history}\n\n"
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
         "- If villain type is known (e.g., tight, loose, aggressive), adjust the recommended strategy accordingly.\n"
         "- Use conversation history to refine recommendations based on prior user interactions.\n\n"
         "Deliver a clear, logical analysis based solely on the provided scenario and history. Close with a bullet-point summary of recommended action(s) and frequencies."
         "Always directly address every part of the user's question, especially when multiple hands, actions, or comparisons are involved."
        )
    ])
    solver_llm = llm(0.1)

    def node_solve(state: ScenarioState) -> Dict[str, Any]:
        logger.info(f"Executing solve node with user_input: {state.user_input}")
        # Load conversation history
        history = state.chat_history or []
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        result = solver_llm.invoke(solve_prompt.format_messages(scenario=state.user_input, chat_history=history_str)).content
        # Save to memory
        memory.save_context(
            inputs={"human": state.user_input},
            outputs={"ai": result}
        )
        # Convert memory's chat_history to list of dicts
        updated_history = [convert_message_to_dict(msg) for msg in memory.load_memory_variables({})["chat_history"]]
        return {"solver_result": result, "chat_history": updated_history}

    respond_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You're **Ray**—the poker buddy who keeps it real. You take even complex solver results and explain them like it's no big deal.\n\n"
         "Here’s your job:\n"
         "• Summarize the solver_result in **1–2 punchy sentences**.\n"
         "• Speak like a confident, chill friend who knows their poker.\n"
         "• Incorporate conversation history to make responses consistent and context-aware.\n\n"
         "**Conversation History:**\n{chat_history}\n\n"
         "**Your tone:**\n"
         "- 🃏 Confident but approachable\n"
         "- ✂️ Concise and action-driven\n"
         "- 🎯 No fluff, just clarity\n"
         "- �.Elements of Poker Fun:** Keep poker fun, sharp, and empowering. Your summary should make even a confusing spot feel like a smart, confident decision."
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
        # Save response to memory
        memory.save_context(
            inputs={"human": state.solver_result},
            outputs={"ai": reply}
        )
        # Save question and answer to database
        db_manager.save_question_answer(state.user_input, reply)
        # Convert memory's chat_history to list of dicts
        updated_history = [convert_message_to_dict(msg) for msg in memory.load_memory_variables({})["chat_history"]]
        return {"response": reply, "chat_history": updated_history}

    logger.info("Building StateGraph")
    g = StateGraph(ScenarioState)
    g.add_node("solve", node_solve)
    g.add_node("respond", node_respond)
    g.set_entry_point("solve")
    g.add_edge("solve", "respond")
    g.add_edge("respond", END)
    return g.compile()

def main():
    graph = build_poker_solver_graph()

    print("Welcome to Poker Coach!")
    while True:
        user_input = input("Describe your poker scenario (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        # Convert memory's chat_history to list of dicts
        memory_history = memory.load_memory_variables({})["chat_history"]
        state_history = [convert_message_to_dict(msg) for msg in memory_history]
        state_input = {
            "user_input": user_input,
            "chat_history": state_history
        }

        result = graph.invoke(state_input)

        print("\n========= FINAL RECOMMENDATION =========")
        print(result["response"])
        print("\n========= CONVERSATION HISTORY =========")
        for msg in result["chat_history"]:
            print(f"{msg['role'].capitalize()}: {msg['content']}")
        print("\n========= STORED QUESTIONS =========")
        for qid, question, answer, timestamp in db_manager.get_all_questions():
            print(f"ID: {qid}, Timestamp: {timestamp}")
            print(f"Question: {question}")
            print(f"Answer: {answer}\n")

if __name__ == "__main__":
    main()