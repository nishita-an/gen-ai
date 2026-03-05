import streamlit as st
import pandas as pd
import json

from llm_clients import call_deepseek, call_qwen, call_groq
from cost_calculator import estimate_cost
from evaluator import evaluate_response

# Set page to wide mode to give the table more room
st.set_page_config(layout="wide")

st.title("LLM Benchmark Dashboard")

prompt = st.text_area("Enter Prompt", height=150)
criteria = st.text_input("Evaluation Criteria", "accuracy, brevity, tone")

if st.button("Run Benchmark", type="primary"):
    results = []
    # Note: Ensure these functions are returning the expected dictionaries
    models = [call_qwen, call_deepseek, call_groq]

    # Create a placeholder to show progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, model_func in enumerate(models):
        status_text.text(f"Running inference for model {idx+1}/{len(models)}...")
        
        result = model_func(prompt)

        cost = estimate_cost(
            result["model"],
            result["input_tokens"],
            result["output_tokens"]
        )

        try:
            status_text.text(f"Evaluating {result['model']}...")
            evaluation = evaluate_response(
                prompt,
                result["response"],
                criteria
            )
            # Handle both string and dict returns from evaluator
            scores = json.loads(evaluation) if isinstance(evaluation, str) else evaluation
        except Exception as e:
            st.error(f"Evaluation failed for {result['model']}: {e}")
            scores = {"accuracy": 0, "brevity": 0, "tone": 0, "overall": 0}

        results.append({
            "Model": result["model"],
            "Latency (s)": round(result["latency"], 2),
            "Estimated Cost ($)": f"{cost:.6f}",
            "Accuracy": scores.get("accuracy", 0),
            "Brevity": scores.get("brevity", 0),
            "Tone": scores.get("tone", 0),
            "Overall": scores.get("overall", 0),
            "Response": result["response"]
        })
        progress_bar.progress((idx + 1) / len(models))

    status_text.empty()
    progress_bar.empty()

    # --- THE PART THAT MAKES IT BIG ---
    st.subheader("Benchmark Results")
    df = pd.DataFrame(results)

    # 1. Use st.dataframe with custom height and full width
    # 2. Add styling to highlight the highest 'Overall' score
    st.dataframe(
        df.style.highlight_max(axis=0, subset=['Overall'], color='#1d4ed8'), 
        use_container_width=True, 
        height=400  # Adjust this number to make it taller
    )

    # Optional: Display full responses in expanders below the table
    st.divider()
    st.subheader("Full Responses")
    for res in results:
        with st.expander(f"View full response from {res['Model']}"):
            st.write(res["Response"])