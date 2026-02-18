from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load models
min_model = joblib.load("retail_min_model.pkl")
max_model = joblib.load("retail_max_model.pkl")

LAST_LAG1 = 40
LAST_LAG2 = 42


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    date = pd.to_datetime(data["date"])

    input_df = pd.DataFrame([{
        "day": date.day,
        "month": date.month,
        "week": int(date.isocalendar().week),
        "is_weekend": int(date.weekday() >= 5),
        "retail_avg_lag1": LAST_LAG1,
        "retail_avg_lag2": LAST_LAG2
    }])

    min_price = min_model.predict(input_df)[0]
    max_price = max_model.predict(input_df)[0]

    return jsonify({
        "retail_min": round(min_price, 2),
        "retail_max": round(max_price, 2)
    })


@app.route("/chatbot", methods=["POST"])
def chatbot_api():
    msg = request.json.get("message", "").lower()

    last_min = 36
    last_max = 41

    if "price" in msg or "today" in msg:
        reply = f"Predicted price is ₹{last_min} – ₹{last_max} per kg."
    elif "earn" in msg or "profit" in msg:
        reply = "Enter quantity in the main screen to see estimated earnings."
    elif "best day" in msg or "sell" in msg:
        reply = "Higher retail max price days are better for selling."
    elif "accuracy" in msg:
        reply = "₹4–₹5 variation is normal due to market demand and supply."
    else:
        reply = "Ask me about price, profit, best day, or accuracy."

    return jsonify({"reply": reply})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
