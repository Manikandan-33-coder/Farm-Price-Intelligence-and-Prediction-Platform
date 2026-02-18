from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load models
min_model = joblib.load("retail_min_model.pkl")
max_model = joblib.load("retail_max_model.pkl")

# Example last known values (later replace with DB)
LAST_LAG1 = 40
LAST_LAG2 = 42
@app.route("/chatbot", methods=["POST"])
def chatbot():
    msg = request.json["message"].lower()

    if "price" in msg:
        reply = "Prices are predicted using historical Dindigul market data."

    elif "accuracy" in msg:
        reply = "₹4–₹5 variation is normal due to demand and supply."

    elif "best day" in msg:
        reply = "Choose days with higher predicted retail max price."

    elif "how" in msg:
        reply = "Select date, enter quantity, and check market price."

    else:
        reply = "Ask me about prices, accuracy, or how to use this app."

    return jsonify({"reply": reply})


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
        "week": date.isocalendar().week,
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

if __name__ == "__main__":
    app.run(debug=True)

