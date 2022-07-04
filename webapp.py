
@app.route("/something", methods=["POST", "GET"])
def process_img():
    data = request.get_data()
    print(data)
    return "hi"

@app.route("/")
def index():
    return "Hello World Test"

if __name__ == "__main__":
    app.run(debug=True, threaded=True, port = 8000)