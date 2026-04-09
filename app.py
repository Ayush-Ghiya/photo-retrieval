# To run this app, use the command: `flask run`

from flask import Flask, jsonify, request

from search import search_images

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"



@app.route('/search', methods=['GET'])
def get_images():
    query = request.args.get('prompt', '')
    
    # Your existing search query logic here
    image_list, results, scores = search_images(prompt=query)  # your function
    
    return jsonify({"images": image_list})