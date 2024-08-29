import base64
import os
from uuid import uuid4
import json

import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession, Part, Content

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional

from utils.retriever import similarity_search
from utils.image import gemini_image_description
from utils.templates import get_prompt_template

import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Vertex AI once for the application
vertexai.init(project="ck-vertex", location="us-central1")

# In-memory store for chat sessions
global chat_sessions
chat_sessions = {}

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = ""
    claim: Optional[dict] = ""
    image_description: Optional[str] = None
    messages: List[Message]

class ChatResponse(BaseModel):
    session_id: str
    role: str
    content: str
    rag: Optional[str]

@app.get("/")
async def main(request: Request):
    return JSONResponse(content={"Available APIs": ["/chat"]}, status_code=200)

# Helper function for the Chat endpoint
def get_chat_session(session_id: str, claim_str: str) -> ChatSession:

    logger.info(f"CHAT SESSIONS: {chat_sessions}")

    if session_id not in chat_sessions:
        # Set the context for each Chat session
        system_context_template = get_prompt_template("system_context")
        system_context_prompt = system_context_template.format(claim=claim_str)

        # Initialize the model, create a ChatSession object, and add the session to our in-memory store
        model = GenerativeModel("gemini-1.5-flash-001", system_instruction=[system_context_prompt])
        chat = model.start_chat()
        chat_sessions[session_id] = chat

        logger.info(f"CHAT SESSION ADDED: {session_id}")

    return chat_sessions[session_id]

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    """
        POST body
        {
            "session_id": "" OR "11111111-1111-1111-1111-111111111111",
            "claim": {"id": 123, "loss_description": "AAAAAAA", ...},
            "image_description": "BBBBBBB",
            "messages": [
                {
                    "role": "user",
                    "content": "XXXXXXX"
                },
                {
                    "role": "model",
                    "content": "YYYYYYY"
                },
                {
                    "role": "user",
                    "content": "ZZZZZZZ"
                }
            ]
        }
    """

    #### Gather the variables from the POST request #######
    claim = chat_request.claim
    claim_str = json.dumps(claim, indent=2)

    image_description = chat_request.image_description

    messages = chat_request.messages
    message = messages[-1].content
    logger.info(f"MESSAGE: {message}")

    # Grab the session_id (if sent in the POST body) OR generate a new one
    session_id = chat_request.session_id if chat_request.session_id else str(uuid4())

    try:
        # Parse documents, including the warranty policies, to ground the answers with retrieved context
        rag_response = similarity_search(prompt=message)
        logger.info(f"RAG CONTEXT: {rag_response}")
    except Exception as e:
        logger.error(f"Error generating RAG response: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

    # Get or create chat session
    chat = get_chat_session(session_id, claim_str)

    # Template and prompt for Gemini multi-turn chat
    chat_template = get_prompt_template("chat")
    chat_prompt = chat_template.format(rag_context=rag_response, image_description=image_description, message=message)

    try:
        parameters = {
            "max_output_tokens": 8192,
            "temperature": 0.5,
            "top_p": 0.8,
            "top_k": 40,
            "candidate_count": 1,
        }

        model_response = chat.send_message(chat_prompt, generation_config=parameters)

        response = ChatResponse(
            session_id=session_id,
            role="model",
            content=model_response.text,
            rag=rag_response
        )

        return JSONResponse(content=response.model_dump(), status_code=200)
    except Exception as e:
        logger.error(f"Error in chat response: {e}")
        raise HTTPException(status_code=500, detail="Request to Gemini's chat feature failed.")

@app.post("/image")
async def process_image(request: Request):

    request_json = await request.json()

    if "image" in request_json:
        base64_image = request_json['image']

    inference = gemini_image_description(base64_image)

    payload = {'image_description': inference.get("description")}
    return JSONResponse(content=payload, status_code=inference.get("status_code"))


if __name__ == "__main__":
  port = int(os.environ.get('PORT', 8080))
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=port)
