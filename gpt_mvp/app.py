import streamlit as st
import os
from dotenv import load_dotenv
from clause_parser import ClauseParser
from gpt_clauses import GPTClauseAnalyzer
import pandas as pd
import json

# Load environment variables (API key)
load_dotenv()

# Page configuration
st.set_page_config(page_title="Legal Clause Risk Analyzer", page_icon="⚖️", layout="wide")

# Initialize parser and analyzer
parser = ClauseParser()

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    if not api_key:
        st.warning("Please enter your OpenAI API key to continue.")
        st.stop()

analyzer = GPTClauseAnalyzer(api_key=api_key)

# App title and description
st.title("Legal Clause Risk Analyzer")
st.markdown("""
Upload a contract document to identify and analyze legal clauses for potential risks.
""")

# File uploader
uploaded_file = st.file_uploader("Upload a contract document (.txt or .docx)", type=["txt", "docx"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_contract." + uploaded_file.name.split('.')[-1], "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    file_path = "temp_contract." + uploaded_file.name.split('.')[-1]
    
    # Parse the file into clauses
    try:
        with st.spinner("Parsing document into clauses..."):
            clauses = parser.parse_file(file_path)
        
        st.success(f"Successfully extracted {len(clauses)} clauses from the document.")
        
        # Let user select which clauses to analyze
        st.subheader("Select clauses to analyze")
        
        # Add a "Select All" checkbox
        select_all = st.checkbox("Select All")
        
        # Display clauses with checkboxes
        selected_clauses = {}
        for i, clause in enumerate(clauses):
            # Truncate display text if too long
            display_text = clause if len(clause) < 100 else clause[:97] + "..."
            selected = st.checkbox(f"Clause {i+1}: {display_text}", value=select_all)
            if selected:
                selected_clauses[i] = clause
        
        # Analyze button
        if st.button("Analyze Selected Clauses") and selected_clauses:
            # Initialize results container
            results = []
            
            # Process each selected clause
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (i, clause) in enumerate(selected_clauses.items()):
                status_text.text(f"Analyzing clause {i+1}...")
                with st.spinner(f"Analyzing clause {i+1}..."):
                    analysis = analyzer.analyze_clause(clause)
                    results.append({
                        "clause_number": i+1,
                        "clause_text": clause,
                        "type": analysis.get("type", "Unknown"),
                        "risk_score": analysis.get("risk_score", 0.0),
                        "explanation": analysis.get("explanation", "No explanation provided")
                    })
                progress_bar.progress((idx + 1) / len(selected_clauses))
            
            status_text.text("Analysis complete!")
            
            # Display results
            st.subheader("Analysis Results")
            
            # Convert to dataframe for display
            df = pd.DataFrame(results)
            
            # Display summary table
            summary_df = df[["clause_number", "type", "risk_score"]].copy()
            summary_df = summary_df.sort_values(by="risk_score", ascending=False)
            
            # Add risk level column
            def risk_category(score):
                if score <= 0.3:
                    return "Low"
                elif score <= 0.6:
                    return "Medium"
                elif score <= 0.9:
                    return "High"
                else:
                    return "Very High"
            
            summary_df["risk_level"] = summary_df["risk_score"].apply(risk_category)
            
            # Display summary
            st.write("Clauses by Risk Level:")
            st.dataframe(summary_df)
            
            # Display detailed results in expandable sections
            st.subheader("Detailed Analysis")
            
            # Sort by risk score (highest first)
            sorted_results = sorted(results, key=lambda x: x["risk_score"], reverse=True)
            
            for result in sorted_results:
                with st.expander(f"Clause {result['clause_number']}: {result['type']} (Risk: {result['risk_score']:.2f})"):
                    st.markdown(f"**Clause Text:**")
                    st.text(result["clause_text"])
                    
                    st.markdown(f"**Type:** {result['type']}")
                    
                    # Display risk score with color
                    risk_score = result["risk_score"]
                    if risk_score <= 0.3:
                        st.markdown(f"**Risk Score:** :green[{risk_score:.2f}] (Low)")
                    elif risk_score <= 0.6:
                        st.markdown(f"**Risk Score:** :orange[{risk_score:.2f}] (Medium)")
                    elif risk_score <= 0.9:
                        st.markdown(f"**Risk Score:** :red[{risk_score:.2f}] (High)")
                    else:
                        st.markdown(f"**Risk Score:** :red[{risk_score:.2f}] (Very High)")
                    
                    st.markdown(f"**Explanation:**")
                    st.write(result["explanation"])
            
            # Option to download results as JSON
            results_json = json.dumps(results, indent=2)
            st.download_button(
                label="Download Analysis as JSON",
                data=results_json,
                file_name="legal_analysis.json",
                mime="application/json"
            )
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
    
    # Clean up the temp file
    try:
        os.remove(file_path)
    except:
        pass

# Add information about API usage in the sidebar
with st.sidebar:
    st.subheader("About")
    st.markdown("""    
    This app uses GPT-4o to analyze legal clauses and assess risk levels. 
    It requires an OpenAI API key and will make API calls for each analyzed clause.
    """)
    
    st.subheader("Risk Levels")
    st.markdown("""
    - **Low (0.0-0.3)**: Standard, balanced clause
    - **Medium (0.4-0.6)**: Some concerning language
    - **High (0.7-0.9)**: Significant one-sided terms
    - **Very High (1.0)**: Potentially unenforceable
    """)
