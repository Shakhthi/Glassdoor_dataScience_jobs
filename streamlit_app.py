import streamlit as st
import requests
import streamlit.components.v1 as components

st.set_page_config(page_title="DS Jobs — Streamlit UI", layout="wide")

st.title("DS Jobs — Streamlit frontend (uses FastAPI)")

st.markdown(
    "This app uploads a CSV to your FastAPI `/predict` endpoint and renders the returned HTML table.\n"
    "Make sure your FastAPI app is running (default: http://localhost:8080)."
)

api_url = st.text_input("FastAPI base URL", value="http://127.0.0.1:8080")

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader("Upload CSV for prediction", type=["csv"])

with col2:
    if st.button("Trigger training"):
        try:
            with st.spinner("Triggering training on FastAPI..."):
                resp = requests.get(f"{api_url}/train", timeout=300)
            if resp.status_code == 200:
                st.success("Training triggered successfully.")
                st.text(resp.text)
            else:
                st.error(f"Train returned {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Train request failed: {e}")

st.divider()

if st.button("Predict"):
    if not uploaded_file:
        st.warning("Please upload a CSV file first.")
    else:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        try:
            with st.spinner("Sending file to FastAPI `/predict`..."):
                resp = requests.post(f"{api_url}/predict", files=files, timeout=120)

            if resp.status_code == 200:
                st.success("Prediction completed — rendering result below.")
                # FastAPI returns a full HTML page (table.html). Render it inside Streamlit.
                components.html(resp.text, height=700, scrolling=True)
            else:
                st.error(f"Predict returned {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Predict request failed: {e}")

st.markdown("---")

st.header("Server output CSV (optional)")
st.markdown(
    "If your FastAPI saves `prediction_output/output.csv` and serves it, try fetching it here.\n"
    "This may 404 unless you configure FastAPI to serve the `prediction_output` folder as static files."
)

if st.button("Fetch server output CSV"):
    try:
        with st.spinner("Fetching /prediction_output/output.csv..."):
            resp = requests.get(f"{api_url}/prediction_output/output.csv", timeout=30)
        if resp.status_code == 200:
            st.download_button("Download output.csv", data=resp.content, file_name="output.csv")
        else:
            st.error(f"Fetch failed: {resp.status_code}")
    except Exception as e:
        st.error(f"Fetch request failed: {e}")

st.markdown("---")

st.info("Run this with: `streamlit run streamlit_app.py`. Ensure FastAPI is running first.")
