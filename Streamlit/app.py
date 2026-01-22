import streamlit as st
import requests

API_URL = "http://localhost:8000/ask"

st.set_page_config(page_title="Customer Intelligence Platform", layout="wide")

st.title("Customer Intelligence Platform")
st.caption(
    "Answers are generated strictly from indexed customer complaint data. "
    "Out-of-scope questions will not be answered."
)

with st.sidebar:
    st.header("About")
    st.write(
        "This tool retrieves relevant customer complaint records and generates "
        "grounded, source-backed answers for analysis."
    )

    st.header("Limitations")
    st.write(
        "- Limited to complaint dataset scope\n"
        "- Retrieval is semantic, not keyword-based\n"
        "- No response for out-of-domain queries"
    )

question = st.text_input(
    "Enter your question",
    placeholder="Why was the complaint escalated?"
)

with st.expander("Advanced settings"):
    k = st.slider(
        "Context breadth",
        min_value=2,
        max_value=10,
        value=4,
        help="Higher values retrieve more context but may add noise and slow responses."
    )

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        payload = {"question": question, "k": k}

        with st.spinner("Retrieving relevant complaints..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=30)

                if response.status_code != 200:
                    st.error(f"API error: {response.status_code}")
                    st.stop()

                data = response.json()

            except requests.exceptions.RequestException:
                st.error("Could not connect to API. Is FastAPI running?")
                st.stop()

        st.subheader("Answer")
        st.markdown(data["answer"])

        if data.get("sources"):
            st.subheader("Sources")
            for src in data["sources"]:
                st.write(f"- {src}")

            st.caption(f"Retrieved {len(data['sources'])} relevant complaint chunks.")
        else:
            st.info(
                "No relevant complaint records were found. "
                "This question may be outside the dataset scope."
            )
