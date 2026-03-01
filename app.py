import streamlit as st
import tiktoken
import random
import html

st.set_page_config(page_title="LLM Tokenizer Visualizer", layout="wide")

st.title("🔎 LLM Tokenizer Visualizer")
st.markdown("Understand how models actually *see* your text.")

# -----------------------------
# Model Selection
# -----------------------------
model_name = st.selectbox(
    "Choose Tokenizer Model",
    ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]
)

encoding = tiktoken.encoding_for_model(model_name)

# -----------------------------
# Text Input
# -----------------------------
user_input = st.text_area(
    "Enter text (paragraph, tweet, code snippet, anything):",
    height=200
)

if user_input:

    # -----------------------------
    # Tokenization
    # -----------------------------
    tokens = encoding.encode(user_input)
    decoded_tokens = [encoding.decode([token]) for token in tokens]

    token_count = len(tokens)
    char_count = len(user_input)
    ratio = round(char_count / token_count, 2) if token_count > 0 else 0

    st.subheader("📊 Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Characters", char_count)
    col2.metric("Tokens", token_count)
    col3.metric("Char / Token Ratio", ratio)

    # -----------------------------
    # Cost Estimation (example pricing)
    # -----------------------------
    cost_per_1k_tokens = 0.002  # Example price
    estimated_cost = (token_count / 1000) * cost_per_1k_tokens

    st.info(f"Estimated Cost (@$0.002 / 1K tokens): ${estimated_cost:.6f}")

    # -----------------------------
    # Color-coded Token Visualization
    # -----------------------------
    st.subheader("🎨 Token Visualization")

    html_tokens = ""
    for token_text in decoded_tokens:
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        safe_text = html.escape(token_text)
        html_tokens += f"""
        <span style="
            background-color:{color};
            padding:4px;
            margin:2px;
            border-radius:4px;
            display:inline-block;
        ">
            {safe_text}
        </span>
        """

    st.markdown(html_tokens, unsafe_allow_html=True)

    # -----------------------------
    # Detailed Token Breakdown
    # -----------------------------
    st.subheader("🧩 Token Breakdown")

    token_data = [
        {
            "Token Index": i,
            "Token ID": token_id,
            "Decoded Text": repr(decoded_tokens[i])
        }
        for i, token_id in enumerate(tokens)
    ]

    st.dataframe(token_data, use_container_width=True)