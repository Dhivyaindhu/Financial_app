import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model_path = "financial_ai_model"   # folder name
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=200
    )

    return generator

generator = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("💰 SME Financial Health AI Advisor")
st.write("AI-powered financial analysis for small businesses")

# -----------------------------
# User Inputs
# -----------------------------
sales = st.number_input("Total Sales", min_value=0.0)
profit = st.number_input("Total Profit", min_value=-1000000.0)
debt_ratio = st.slider("Debt to Income Ratio", 0.0, 1.0, 0.3)
credit_score = st.slider("Credit Score", 300, 900, 700)

# -----------------------------
# Generate AI Prompt
# -----------------------------
if st.button("Analyze Financial Health"):

    profit_margin = profit / sales if sales > 0 else 0

    user_prompt = f"""
    SME Financial Summary:
    Sales: {sales}
    Profit Margin: {profit_margin}
    Debt to Income Ratio: {debt_ratio}
    Credit Score: {credit_score}

    Provide financial recommendations.
    """

    with st.spinner("Analyzing financial data..."):

        result = generator(user_prompt)[0]["generated_text"]

    st.subheader("📊 AI Financial Recommendation")
    st.write(result)
