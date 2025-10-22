"""
Plant Maintenance Server
Optimized FastAPI server with clean, human-written code
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Import our optimized components
from question_parser import QuestionParser
from data_analyzer import DataAnalyzer
from answer_formatter import AnswerFormatter

# FastAPI Setup
app = FastAPI(title="Plant Maintenance Q&A System", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Pydantic Models
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    analysis_type: Optional[str] = None
    data: Optional[dict] = None
    timestamp: str

# Global Components
question_parser = None
data_analyzer = None
answer_formatter = None

def load_components():
    """Load all system components."""
    global question_parser, data_analyzer, answer_formatter
    
    try:
        question_parser = QuestionParser()
        data_analyzer = DataAnalyzer("plant_dataset.csv")
        answer_formatter = AnswerFormatter()
        print("✅ Plant Maintenance system loaded successfully")
    except Exception as e:
        print(f"❌ Error loading components: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load components on startup."""
    load_components()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Plant Maintenance Q&A System",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "question_parser": question_parser is not None,
            "data_analyzer": data_analyzer is not None,
            "answer_formatter": answer_formatter is not None
        }
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Main endpoint for asking questions."""
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        print(f"🔍 Processing: {question}")
        
        # Parse question
        parsed = question_parser.parse_question(question)
        if not parsed.get("success"):
            raise HTTPException(status_code=400, detail=f"Question parsing failed: {parsed.get('error')}")
        
        # Analyze data
        analysis = data_analyzer.analyze_data(parsed)
        if not analysis.get("success"):
            raise HTTPException(status_code=500, detail=f"Data analysis failed: {analysis.get('error')}")
        
        # Format answer
        answer = answer_formatter.format_answer(analysis, parsed)
        
        return QuestionResponse(
            answer=answer,
            analysis_type=parsed.get("analysis_type"),
            data=analysis,
            timestamp=__import__("datetime").datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/examples")
async def get_examples():
    """Get example questions."""
    return {
        "examples": [
            "How many pumps are there?",
            "What is the average temperature?",
            "Which machines are outdated?",
            "Show me high risk equipment",
            "What's the highest vibration?",
            "Count all motors"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
