from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from datetime import datetime

from question_parser import QuestionParser
from data_analyzer import DataAnalyzer
from answer_formatter import AnswerFormatter

app = FastAPI(title="Plant Maintenance Q&A System", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    analysis_type: Optional[str] = None
    data: Optional[dict] = None
    timestamp: str

question_parser = None
data_analyzer = None
answer_formatter = None

def load_components():
    global question_parser, data_analyzer, answer_formatter
    
    try:
        question_parser = QuestionParser()
        data_analyzer = DataAnalyzer("plant_dataset.csv")
        answer_formatter = AnswerFormatter()
    except Exception as e:
        raise

@app.on_event("startup")
async def startup_event():
    load_components()

@app.get("/")
async def root():
    return {
        "message": "Plant Maintenance Q&A System",
        "status": "running",
        "version": "1.0.0"
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        parsed = question_parser.parse_question(question)
        if not parsed.get("success"):
            raise HTTPException(status_code=400, detail=f"Question parsing failed: {parsed.get('error')}")
        
        analysis = data_analyzer.analyze_data(parsed)
        if not analysis.get("success"):
            raise HTTPException(status_code=500, detail=f"Data analysis failed: {analysis.get('error')}")
        
        answer = answer_formatter.format_answer(analysis, parsed)
        
        return QuestionResponse(
            answer=answer,
            analysis_type=parsed.get("analysis_type"),
            data=analysis,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
