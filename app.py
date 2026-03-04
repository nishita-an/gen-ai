import streamlit as st
import pandas as pd
import json

from llm_clients import call_deepseek, call_qwen, call_groq
from cost_calculator import estimate_cost
from evaluator import evaluate_response

st.title("LLM Benchmark Dashboard")

prompt = st.text_area("Enter Prompt")
criteria = st.text_input("Evaluation Criteria", "accuracy, brevity, tone")

if st.button("Run Benchmark"):

    results = []
    models = [call_qwen, call_deepseek, call_groq]

    for model_func in models:
        result = model_func(prompt)

        cost = estimate_cost(
            result["model"],
            result["input_tokens"],
            result["output_tokens"]
        )

        try:
            evaluation = evaluate_response(
                prompt,
                result["response"],
                criteria
            )
            scores = json.loads(evaluation)
        except Exception as e:
            st.error(f"Evaluation failed for {result['model']}: {e}")
            scores = {"accuracy": 0, "brevity": 0, "tone": 0, "overall": 0}

        results.append({
            "Model": result["model"],
            "Latency (s)": round(result["latency"], 2),
            "Estimated Cost ($)": round(cost, 6),
            "Accuracy": scores["accuracy"],
            "Brevity": scores["brevity"],
            "Tone": scores["tone"],
            "Overall Score": scores["overall"],
            "Response": result["response"]
        })

    df = pd.DataFrame(results)
    st.dataframe(df)