from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html", results="")

@app.route("/predict", methods=["POST"])
def predict_datapoint():
    try:
        carat = float(request.form["carat"])
        depth = float(request.form["depth"])
        table = float(request.form["table"])
        x = float(request.form["x"])
        y = float(request.form["y"])
        z = float(request.form["z"])

        cut = request.form["cut"]
        color = request.form["color"]
        clarity = request.form["clarity"]

        predicted_price = round(
            (carat * 5000) + ((x + y + z) * 100) + (depth * 10),
            2
        )

        return render_template("home.html", results=f"₹ {predicted_price}")

    except Exception as e:
        return render_template("home.html", results=f"Invalid input")

if __name__ == "__main__":
    app.run(debug=True)
