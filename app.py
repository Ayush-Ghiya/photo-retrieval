# To run this app, use the command: `flask run`

from flask import Flask, jsonify, request, send_file

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

#TEMPORARY Serve images from the cifar_images directory
@app.route('/images/<filename>', methods=['GET'])
def serve_image(filename):
    image_path = f"./cifar_images/{filename}"
    return send_file(image_path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)