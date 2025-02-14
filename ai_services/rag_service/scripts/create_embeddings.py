import os
from langchain_community.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# 1. Φορτώνουμε τα αρχεία από τον φάκελο data/
data_folder = "ai_services/rag_service/data"
documents = []
for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join(data_folder, filename))
        documents.extend(loader.load())

# 2. Μετατρέπουμε τα κείμενα σε embeddings
embedding_model = OpenAIEmbeddings()
vector_db = FAISS.from_documents(documents, embedding_model)

# 3. Αποθηκεύουμε τα embeddings στη βάση
vector_db.save_local("ai_services/rag_service/embeddings/knowledge_base")

print("✅ Embeddings created & saved successfully!")
