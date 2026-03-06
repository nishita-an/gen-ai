import streamlit as st
import pandas as pd
import json
# Updated imports to match the new llm_clients.py functions
from llm_clients import call_gpt_oss, call_llama, call_qwen
from cost_calculator import estimate_cost
from evaluator import evaluate_response

st.set_page_config(page_title="LLM Benchmarker", layout="wide")

st.title("🚀 LLM Benchmark Dashboard")
st.markdown("Compare performance, cost, and quality across GPT-OSS, Llama 3.3, and Qwen 3 on Groq.")

# UI Layout
col1, col2 = st.columns([2, 1])
with col1:
    prompt = st.text_area("Enter Test Prompt", height=150, placeholder="e.g. Write a Python script to scrape a website...")
with col2:
    criteria = st.text_input("Evaluation Criteria", "Technical accuracy, code efficiency, helpful tone")
    run_btn = st.button("Run Benchmark", type="primary", use_container_width=True)

if run_btn:
    results = []
    models = [call_gpt_oss, call_llama, call_qwen]
    
    status = st.status("Running models...", expanded=True)
    
    for model_func in models:
        # 1. Inference
        res = model_func(prompt)
        status.write(f"✅ Received response from {res['model']}")
        
        # 2. Cost Calculation
        cost = estimate_cost(res["model"], res["input_tokens"], res["output_tokens"])
        
        # 3. Evaluation
        scores = evaluate_response(prompt, res["response"], criteria)
        status.write(f"⚖️ Evaluated {res['model']}")

        results.append({
            "Model": res["model"],
            "Latency (s)": round(res["latency"], 2),
            "Cost ($)": f"{cost:.6f}",
            "Accuracy": scores.get("accuracy", 0),
            "Brevity": scores.get("brevity", 0),
            "Tone": scores.get("tone", 0),
            "Overall": scores.get("overall", 0),
            "Reasoning": scores.get("reasoning", "N/A"),
            "Full Response": res["response"]
        })
    
    status.update(label="Benchmark Complete!", state="complete", expanded=False)

    # --- LARGE DISPLAY TABLE ---
    st.subheader("Leaderboard")
    df = pd.DataFrame(results)
    
    # Stylized DataFrame
    st.dataframe(
        df.drop(columns=["Full Response"]).style.highlight_max(axis=0, subset=['Overall'], color='#004d00'),
        use_container_width=True,
        height=300
    )

    # --- DETAILED RESPONSES ---
    st.divider()
    st.subheader("Detailed Model Outputs")
    for r in results:
        with st.expander(f"View Output: {r['Model']} (Score: {r['Overall']}/10)"):
            st.info(f"**Evaluator Reasoning:** {r['Reasoning']}")
            st.code(r["Full Response"], language="markdown")