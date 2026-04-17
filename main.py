from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from routes.chat import router as chat_router

app = FastAPI(title="First Aid Chatbot")
 
app.include_router(chat_router) 

@app.get("/health")
def health():
    return {"status": "ok"}
