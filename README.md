

# 🚀 LLM Benchmark Dashboard

A real-time benchmarking tool built with **Streamlit** and **Groq** to compare performance, cost, and quality across multiple Large Language Models (LLMs). This dashboard evaluates models on latency, token costs, and qualitative accuracy using an automated AI judge.

## 📊 Features

* **Multi-Model Inference:** Supports GPT-OSS-120B, Llama 3.3-70B, and Qwen-3-32B via Groq.
* **Real-time Metrics:** Tracks end-to-end latency and calculates precise API costs per request.
* **Automated Evaluation:** Uses a "Judge LLM" (Llama 3.3) to score responses based on custom criteria (Accuracy, Brevity, Tone).
* **Fallback Logic:** Includes a robust retry strategy and model fallback if the primary evaluator fails.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Inference Engine:** Groq Cloud API
* **Language:** Python 3.10+
* **Data Handling:** Pandas

---

## ⚙️ Local Setup

1. **Clone the Repository:**
```bash
git clone -b llm_benchmark https://github.com/nishita-an/gen-ai.git
cd gen-ai

```


2. **Create a Virtual Environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


4. **Environment Variables:**
Create a `.env` file in the root directory and add your Groq API Key:
```text
GROQ_API_KEY=your_groq_api_key_here

```


5. **Run the App:**
```bash
streamlit run app.py

```



---

## ☁️ Deployment (AWS EC2)

### 1. Security Group Configuration

Ensure your EC2 instance has **Port 8501** open in the Inbound Rules to allow Streamlit traffic.

### 2. Deployment Steps

1. SSH into your instance: `ssh -i "your-key.pem" ubuntu@<EC2-IP>`.
2. Install system dependencies: `sudo apt update && sudo apt install python3-pip python3-venv -y`.
3. Follow the **Local Setup** steps above on the server.

### 3. Keep-Alive with Screen

To keep the dashboard running after closing your SSH session:

```bash
screen -S llm_benchmark
streamlit run app.py
# Press Ctrl+A then D to detach

```

---

## 📂 Project Structure

* `app.py`: Main Streamlit UI and application logic.
* `llm_clients.py`: API wrappers and retry strategies for Groq.
* `evaluator.py`: Logic for the AI Quality Auditor and scoring rubric.
* `cost_calculator.py`: Token-based pricing logic for supported models.
* `config.py`: Environment variable management.

---

