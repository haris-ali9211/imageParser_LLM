from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import pytesseract
from langchain.docstore.document import Document
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from fastapi.middleware.cors import CORSMiddleware
import io
import os
import requests

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the language model
# local_llm = 'gemma'
# local_llm = 'llama3'
local_llm = 'llama3.1'
# local_llm = 'mistral'

# --------linus
# host_ip = os.environ.get('HOST_IP', 'host.docker.internal')
# llm = ChatOllama(model=local_llm, temperature=0, base_url=f"http://{host_ip}:11434")
# llm = ChatOllama(model=local_llm, temperature=0)

# --------mac
# Load the language model
llm = ChatOllama(model=local_llm, temperature=0, base_url="http://host.docker.internal:11434")

# Define the QA prompt template
qa_system_prompt = """You're a friendly expert in math and finance. Given the context, answer the question, and provide a clear, brief explanation afterward.
Question: {input}
Context: {context}
Answer: assistant"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create the question-answer chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Chat history
chat_history = []

class QueryModel(BaseModel):
    question: str
    image: Optional[UploadFile] = None

@app.post("/analyze-image/")
async def analyze_image(question: str = Form(...), file: UploadFile = File(...)):
    try:
        # Load the image from the uploaded file
        image = Image.open(io.BytesIO(await file.read()))

        # Use Tesseract to extract text from the image
        extracted_text = pytesseract.image_to_string(image)

        # Create a context document from the extracted text
        context_document = Document(page_content=extracted_text)

        # Ask the question
        ai_msg_1 = question_answer_chain.invoke({"input": question, "chat_history": [], "context": [context_document]})
        chat_history.extend([HumanMessage(content=question), ai_msg_1])

        return JSONResponse(content={"answer": ai_msg_1})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
