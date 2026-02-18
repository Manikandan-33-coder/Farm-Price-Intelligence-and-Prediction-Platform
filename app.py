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
#from flask import Flask, request, jsonify

@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_msg = request.json["message"].lower()

    if "what is this app" in user_msg:
        reply = "This app predicts vegetable prices for Dindigul markets."

    elif "accuracy" in user_msg:
        reply = "Predictions are based on historical data. ₹4–₹5 variation is normal."

    elif "best day" in user_msg:
        reply = "Choose days with higher predicted max price."

    elif "how to use" in user_msg:
        reply = "Select date, enter quantity, and click Check Market Price."

    else:
        reply = "I can help with price prediction, usage, and market advice."

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

