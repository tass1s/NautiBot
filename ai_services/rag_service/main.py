from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 1. Φορτώνουμε τα embeddings από τη FAISS database
vector_db = FAISS.load_local("ai_services/rag_service/embeddings/knowledge_base", OpenAIEmbeddings())

# 2. Δημιουργούμε το RAG System
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=vector_db.as_retriever())

# 3. Χρήστης ρωτάει κάτι και το σύστημα απαντά
while True:
    query = input("🔹 Ρώτα κάτι (ή γράψε 'exit' για έξοδο): ")
    if query.lower() == "exit":
        break
    response = qa_chain.run(query)
    print(f"🤖 Απάντηση: {response}\n")
