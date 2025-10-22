# Plant Maintenance Q&A System

A simple system that answers questions about plant equipment data using natural language.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install pandas fastapi uvicorn requests
   ```

2. **Start the server:**
   ```bash
   python plant_server.py
   ```

3. **Ask questions:**
   ```bash
   python ask_questions.py
   ```

## Example Questions

- "How many pumps are there?"
- "What is the average temperature?"
- "Which machines are outdated?"
- "Top 5 equipment with highest vibration level"
- "Show me all motors"

## How It Works

1. **Question Parser** - understands your questions
2. **Data Analyzer** - processes the plant data
3. **Answer Formatter** - gives you clear answers

## Features

- Count equipment (pumps, motors, valves)
- Calculate averages, maximums, minimums
- Find outdated and high-risk equipment
- Handle typos and different question formats
- Fast responses with detailed information

## Files

- `plant_server.py` - Main server
- `question_parser.py` - Question understanding
- `data_analyzer.py` - Data processing
- `answer_formatter.py` - Response formatting
- `ask_questions.py` - Question interface
- `plant_dataset.csv` - Your data

## API

- **POST** `/ask` - Ask questions
- **GET** `/health` - Check system status

That's it! Simple and effective plant maintenance data analysis.