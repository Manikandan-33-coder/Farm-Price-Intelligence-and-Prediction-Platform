from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load ML models
min_model = joblib.load("retail_min_model.pkl")
max_model = joblib.load("retail_max_model.pkl")

# Last known lag values (can be replaced by DB later)
LAST_LAG1 = 40
LAST_LAG2 = 42

# Store last prediction (used by chatbot)
last_prediction = {
    "min": None,
    "max": None
}

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- PRICE PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    date = pd.to_datetime(data["date"])

    input_df = pd.DataFrame([{
        "day": date.day,
        "month": date.month,
        "week": date.isocalendar().week,
        "is_weekend": int(date.weekday() >= 5),
        "retail_avg_lag1": LAST_LAG1,
        "retail_avg_lag2": LAST_LAG2
    }])

    min_price = float(min_model.predict(input_df)[0])
    max_price = float(max_model.predict(input_df)[0])

    # Save for chatbot use
    last_prediction["min"] = round(min_price, 2)
    last_prediction["max"] = round(max_price, 2)

    return jsonify({
        "retail_min": last_prediction["min"],
        "retail_max": last_prediction["max"]
    })


# ---------------- SMART CHATBOT ----------------
@app.route("/chatbot", methods=["POST"])
def chatbot():
    msg = request.json["message"].lower()

    min_p = last_prediction["min"]
    max_p = last_prediction["max"]

    # Default if no prediction yet
    if min_p is None or max_p is None:
        min_p, max_p = 36, 41

    if "price" in msg or "today" in msg:
        reply = f"Predicted price is ₹{min_p} – ₹{max_p} per kg."

    elif "earn" in msg or "profit" in msg:
        reply = "Enter quantity in main screen to calculate your earnings."

    elif "best day" in msg or "sell" in msg:
        if max_p >= 40:
            reply = "Good price. You can take vegetables to market."
        else:
            reply = "Price is low. Waiting may give better returns."

    elif "accuracy" in msg or "correct" in msg:
        reply = "₹4–₹5 variation is normal due to demand and supply."

    elif "how" in msg or "use" in msg:
        reply = "Select date → Enter quantity → Click Check Market Price."

    elif "farmer" in msg:
        reply = "This app is built specially for Dindigul farmers."

    else:
        reply = (
            "I can help you with:\n"
            "• Price information\n"
            "• Earnings calculation\n"
            "• Best selling day\n"
            "• Prediction accuracy"
        )

    return jsonify({"reply": reply})


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)
