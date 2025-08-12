from __future__ import annotations

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, List, Any
from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt
import random
import aiosmtplib
from email.message import EmailMessage
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import os
import logging
from langchain_xai import ChatXAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from database import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT config
SECRET_KEY = "58b1f13a582036ae2007467ceec1f280"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Email config (use real credentials)
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "your-email@gmail.com"
SMTP_PASSWORD = "your-app-password"

# In-memory reset code storage
reset_codes = {}

# OAuth2 password bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Initialize database and memory
db_manager = DatabaseManager()
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# ------------------- Models -------------------

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    confirm_password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    code: str
    new_password: str
    confirm_password: str

class GoogleAuthRequest(BaseModel):
    token: str

class UserInput(BaseModel):
    scenario: str

class ScenarioState(BaseModel):
    user_id: int
    user_input: str
    scenario: Optional[str] = None
    solver_result: Optional[str] = None
    response: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None

# ------------------- Utility Functions -------------------

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def send_verification_code(email: str, code: str):
    message = EmailMessage()
    message["From"] = SMTP_USERNAME
    message["To"] = email
    message["Subject"] = "Password Reset Code"
    message.set_content(f"Your password reset code is: {code}")
    
    await aiosmtplib.send(
        message,
        hostname=SMTP_HOST,
        port=SMTP_PORT,
        start_tls=True,
        username=SMTP_USERNAME,
        password=SMTP_PASSWORD,
    )

# ------------------- Endpoints -------------------

@app.post("/signup")
async def signup(user: UserCreate):
    if user.password != user.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    if len(user.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    hashed_pw = hash_password(user.password)
    success = db_manager.save_user(user.name, user.email, hashed_pw)
    if not success:
        raise HTTPException(status_code=400, detail="Email already registered")
    return {"message": "User registered successfully"}

@app.post("/login", response_model=Token)
async def login(user: UserLogin):
    try:
        db_user = db_manager.get_user_by_email(user.email)
        if not db_user:
            logger.error(f"No user found for email: {user.email}")
            raise HTTPException(status_code=400, detail="Invalid email or password")
        if not verify_password(user.password, db_user["password"]):
            logger.error(f"Password verification failed for email: {user.email}")
            raise HTTPException(status_code=400, detail="Invalid email or password")

        token = create_access_token(data={"sub": user.email, "user_id": db_user["id"]})
        logger.info(f"Login successful for {user.email}")
        return {"access_token": token, "token_type": "bearer"}
    except Exception as e:
        logger.error(f"Login error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
@app.post("/request-password-reset")
async def request_password_reset(email: EmailStr):
    user = db_manager.get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="Email not found")

    code = f"{random.randint(100000, 999999)}"
    reset_codes[email] = code
    await send_verification_code(email, code)
    return {"message": "Verification code sent to email"}

@app.post("/reset-password")
async def reset_password(data: ResetPasswordRequest):
    if data.email not in reset_codes or reset_codes[data.email] != data.code:
        raise HTTPException(status_code=400, detail="Invalid or expired verification code")
    if data.new_password != data.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    if len(data.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password too short")

    hashed_pw = hash_password(data.new_password)
    success = db_manager.update_user_password(data.email, hashed_pw)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to reset password")
    
    del reset_codes[data.email]
    return {"message": "Password reset successful"}

@app.post("/google-signup", response_model=Token)
async def google_signup(data: GoogleAuthRequest):
    try:
        idinfo = id_token.verify_oauth2_token(
            data.token,
            google_requests.Request(),
            audience=None  # You can add your CLIENT_ID here
        )

        email = idinfo.get("email")
        name = idinfo.get("name", "Google User")

        if not email:
            raise HTTPException(status_code=400, detail="Invalid Google token")

        user = db_manager.get_user_by_email(email)
        if not user:
            fake_pw = hash_password("google_auth_user")
            db_manager.save_user(name, email, fake_pw)

        user = db_manager.get_user_by_email(email)
        access_token = create_access_token(data={"sub": email, "user_id": user["id"]})
        return {"access_token": access_token, "token_type": "bearer"}

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Google token")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------- Poker Analysis -------------------

def llm(temp: float = 0.0):
    key = os.getenv("XAI_API_KEY")

    if not key:
        raise ValueError("XAI_API_KEY environment variable is not set.")

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
        history = state.chat_history or []
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        result = solver_llm.invoke(solve_prompt.format_messages(scenario=state.user_input, chat_history=history_str)).content
        memory.save_context(
            inputs={"human": state.user_input},
            outputs={"ai": result}
        )
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
        memory.save_context(
            inputs={"human": state.solver_result},
            outputs={"ai": reply}
        )
        db_manager.save_question_answer(state.user_id, state.user_input, reply)
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

# Build the graph once at startup
graph = build_poker_solver_graph()

@app.post("/analyze-poker-scenario")
async def analyze_poker_scenario(user_input: UserInput, token: str = Depends(oauth2_scheme)):
    if not user_input.scenario:
        raise HTTPException(status_code=400, detail="Scenario cannot be empty")
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_email = payload.get("sub")
        user_id = payload.get("user_id")
        if not user_email or not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    memory_history = memory.load_memory_variables({})["chat_history"]
    state_history = [convert_message_to_dict(msg) for msg in memory_history]
    state_input = {
        "user_id": user_id,
        "user_input": user_input.scenario,
        "chat_history": state_history
    }

    result = graph.invoke(state_input)

    return {
        "recommendation": result["response"],
        "chat_history": result["chat_history"]
    }

@app.get("/stored-questions")
async def get_stored_questions(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    questions = db_manager.get_user_questions(user_id)
    return [
        {
            "id": qid,
            "question": question,
            "answer": answer,
            "timestamp": timestamp
        }
        for qid, question, answer, timestamp in questions
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)