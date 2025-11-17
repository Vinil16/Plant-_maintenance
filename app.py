import streamlit as st
import pandas as pd
from question_parser import QuestionParser
from data_analyzer import DataAnalyzer
from answer_formatter import AnswerFormatter

# Page configuration
st.set_page_config(
    page_title="Plant Maintenance Assistant",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .example-question {
        padding: 0.75rem;
        margin: 0.5rem 0;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .example-question:hover {
        background-color: #e0e4e8;
    }
    .answer-box {
        padding: 1.5rem;
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        border-radius: 0.5rem;
        margin-top: 1rem;
        white-space: pre-wrap;
        color: #333;
    }
    .info-box {
        padding: 1rem;
        background-color: #e7f3ff;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'question_parser' not in st.session_state:
    with st.spinner("Loading Plant Maintenance System..."):
        try:
            st.session_state.question_parser = QuestionParser()
            st.session_state.data_analyzer = DataAnalyzer("plant_dataset.csv")
            st.session_state.answer_formatter = AnswerFormatter()
            st.session_state.loaded = True
        except Exception as e:
            st.error(f"Error loading system: {str(e)}")
            st.session_state.loaded = False

if 'history' not in st.session_state:
    st.session_state.history = []

# Main header
st.markdown('<h1 class="main-header">🔧 Plant Maintenance Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about your equipment, get insights, and manage maintenance schedules</p>', unsafe_allow_html=True)

# Sidebar with examples and info
with st.sidebar:
    st.header("💡 Example Questions")
    
    st.markdown("**Dataset Queries:**")
    dataset_examples = [
        "How many pumps are in the system?",
        "What is the average temperature?",
        "What is the highest pressure?",
        "Show me all transformers",
        "Count compressors",
        "What is the maximum vibration level?"
    ]
    
    for example in dataset_examples:
        if st.button(example, key=f"dataset_{example}", use_container_width=True):
            st.session_state.current_question = example
            st.session_state.process_question = example
            st.rerun()
    
    st.markdown("---")
    st.markdown("**ML Model Predictions:**")
    ml_examples = [
        "Will TRANS-044 fail soon?",
        "Top 10 preventive machines",
        "Top 5 low risk machines",
        "Show me high-risk equipment",
        "Will EXCH-046 fail soon?",
        "Top 15 high risk machines"
    ]
    
    for example in ml_examples:
        if st.button(example, key=f"ml_{example}", use_container_width=True):
            st.session_state.current_question = example
            st.session_state.process_question = example
            st.rerun()
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.info("""
    This system uses machine learning to predict equipment failures and analyze plant data.
    
    You can ask about:
    - Equipment counts and statistics
    - Risk assessments
    - Failure predictions
    - Maintenance schedules
    - Top N machines by risk level
    """)

# Main content area
if not st.session_state.get('loaded', False):
    st.error("System failed to load. Please check the error message above.")
else:
    # Question input
    col1, col2 = st.columns([4, 1])
    
    # Get question from input or from example click
    default_question = st.session_state.get('current_question', '')
    
    with col1:
        question = st.text_input(
            "Ask your question:",
            value=default_question,
            placeholder="e.g., How many pumps are in the system?",
            label_visibility="collapsed",
            key="question_input"
        )
    
    with col2:
        ask_button = st.button("Ask", type="primary", use_container_width=True)
    
    # Determine which question to process
    question_to_process = None
    if 'process_question' in st.session_state:
        # Question from example button click
        question_to_process = st.session_state.process_question
        del st.session_state.process_question
        if 'current_question' in st.session_state:
            del st.session_state.current_question
    elif ask_button and question:
        # Question from manual input
        question_to_process = question
    
    # Process question
    if question_to_process:
        with st.spinner("Analyzing your question..."):
            try:
                # Parse question
                parsed = st.session_state.question_parser.parse_question(question_to_process)
                
                # Check if parsed is None or doesn't have success key
                if parsed is None:
                    st.warning(" I couldn't process that question. Please try rephrasing or use one of the example questions from the sidebar.")
                elif not parsed.get("success", False):
                    error_msg = parsed.get('error', 'Please try rephrasing or use one of the example questions.')
                    st.warning(f" {error_msg}")
                    st.info("**Tip:** Try asking questions like:\n- 'How many pumps are in the system?'\n- 'Will TRANS-044 fail soon?'\n- 'Top 10 preventive machines'")
                else:
                    # Analyze data
                    analysis = st.session_state.data_analyzer.analyze_data(parsed)
                    
                    if analysis is None:
                        st.error("Analysis returned no results. Please try again.")
                    elif not analysis.get("success", False):
                        error_msg = analysis.get('error', 'Unknown error')
                        st.error(f"Analysis failed: {error_msg}")
                    else:
                        # Format answer
                        answer = st.session_state.answer_formatter.format_answer(analysis, parsed)
                        
                        # Add to history
                        st.session_state.history.append({
                            "question": question_to_process,
                            "answer": answer,
                            "type": parsed.get("analysis_type", "unknown")
                        })
                        
                        # Display answer with proper formatting
                        st.markdown("### Answer")
                        # Display in a code block to preserve formatting
                        st.code(answer, language=None)
                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    elif ask_button and not question:
        st.warning("Please enter a question first.")
    
    # Show conversation history
    if st.session_state.history:
        st.markdown("---")
        st.header("📜 Recent Questions")
        
        # Show last 5 questions
        for i, item in enumerate(reversed(st.session_state.history[-5:])):
            with st.expander(f"Q: {item['question']}", expanded=False):
                st.markdown(item['answer'])
        
        # Clear history button
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    
    # Quick stats section
    st.markdown("---")
    st.header("📊 Quick Stats")
    
    try:
        df = pd.read_csv("plant_dataset.csv", encoding="latin-1")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Equipment", len(df))
        
        with col2:
            if 'asset_type' in df.columns:
                unique_types = df['asset_type'].nunique()
                st.metric("Equipment Types", unique_types)
            else:
                st.metric("Equipment Types", "N/A")
        
        with col3:
            if 'temperature' in df.columns:
                avg_temp = df['temperature'].mean()
                st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
            else:
                st.metric("Avg Temperature", "N/A")
        
        with col4:
            if 'vibration_level' in df.columns:
                avg_vib = df['vibration_level'].mean()
                st.metric("Avg Vibration", f"{avg_vib:.2f} mm/s")
            else:
                st.metric("Avg Vibration", "N/A")
                
    except Exception as e:
        st.warning(f"Could not load quick stats: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        Plant Maintenance Assistant | Powered by ML Predictive Analytics
    </div>
    """,
    unsafe_allow_html=True
)

