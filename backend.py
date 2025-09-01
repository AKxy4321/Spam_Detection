import io

from flask import Flask, jsonify, request
from PyPDF2 import PdfReader

from predict_pdf_module import predict_pdf

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    try:
        # Read PDF into memory instead of saving to disk
        pdf_bytes = file.read()
        reader = PdfReader(io.BytesIO(pdf_bytes))

        # Extract text only from the first page
        first_page = reader.pages[0]
        text = first_page.extract_text() or ""

        result = predict_pdf(text)

        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"Failed to read PDF: {e}"}), 500


if __name__ == "__main__":
    print("Starting Backend Server")

    # debug == true slows down the server -- keep it only when you need to debug
    app.run(port=5000, debug=False)
