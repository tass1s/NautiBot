from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

class QueryModel(BaseModel):
    query: str

@app.post("/chat")
def chat_endpoint(query: QueryModel):
    response = requests.post("http://127.0.0.1:8001/chat", json={"query": query.query})
    return response.json()

@app.post("/rag")
def rag_endpoint(query: QueryModel):
    response = requests.post("http://127.0.0.1:8002/rag", json={"query": query.query})
    return response.json()
