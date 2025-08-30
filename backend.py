import os

from flask import Flask, jsonify, request

from predict_pdf_module import predict_pdf

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    temp_path = "temp.pdf"
    file.save(temp_path)

    try:
        result = predict_pdf(temp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Delete the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    print("Starting Backend Server")
    app.run(port=5000, debug=True)
