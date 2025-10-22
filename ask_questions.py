"""
Simple client to ask questions to the plant maintenance system
"""

import requests
import json

def ask_question(question):
    """Ask a question to the plant maintenance system."""
    try:
        response = requests.post(
            "http://localhost:8000/ask",
            json={"question": question},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("answer", "No answer received")
        else:
            return f"ERROR: {response.status_code} - {response.text}"
            
    except requests.exceptions.ConnectionError:
        return "ERROR: Server not running. Start with: python plant_maintenance_system.py server"
    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    """Main function to ask questions."""
    
    print("Plant Maintenance Q&A System")
    print("=" * 50)
    print("Ask questions about your plant data!")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("Server is running")
        else:
            print("Server might not be running properly")
    except:
        print("Server is not running.")
        return
    
    while True:
        question = input("\n Ask question: ").strip()
        
        if question.lower() in ['quit', 'exit']:
            break
        
        if not question:
            continue
        
        print("Processing...")
        answer = ask_question(question)
        print(f"Answer: {answer}")
        print("-" * 50)

if __name__ == "__main__":
    main()