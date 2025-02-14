from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI

app = FastAPI()

chat_model = ChatOpenAI()

@app.post("/chat")
def chat(query: str):
    response = chat_model.predict(query)
    return {"response": response}
