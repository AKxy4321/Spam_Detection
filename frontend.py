import requests
import streamlit as st

st.set_page_config(page_title="PDF Spam Detector", layout="centered")
st.title("📄 PDF Spam Detector")
st.write("Upload a PDF file to predict if it is SPAM or HAM.")

uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file is not None:
    st.info("Uploading PDF to server...")

    files = {"file": (uploaded_file.name, uploaded_file.getbuffer(), "application/pdf")}
    try:
        response = requests.post("http://127.0.0.1:5000/predict", files=files)
        if response.status_code == 200:
            result = response.json()
            if "error" in result:
                st.error(result["error"])
            else:
                label = result["label"].upper()

                # Color + Emoji output
                if label == "SPAM":
                    st.markdown(
                        f"<h3 style='color:red;'>🚫 Prediction: {label}</h3>",
                        unsafe_allow_html=True,
                    )
                elif label == "HAM":
                    st.markdown(
                        f"<h3 style='color:green;'>✅ Prediction: {label}</h3>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning(f"Prediction: {label}")

                st.write(f"Text length: {result['text_length']}")
                st.write(f"Preview of extracted text: {result['preview']}")
        else:
            st.error(f"Server error: {response.text}")
    except Exception as e:
        st.error(f"Failed to connect to server: {e}")
