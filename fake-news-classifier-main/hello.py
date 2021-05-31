from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def hello_world():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        return request.form['article']
