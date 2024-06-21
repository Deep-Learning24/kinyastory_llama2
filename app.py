from flask import Flask, render_template, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.json['prompt']
    llama2_path = "./run"  #./run out/model.bin -z data/tok4096.bin
    try:
        result = subprocess.run([llama2_path, prompt], capture_output=True, text=True, timeout=30)
        generated_text = result.stdout
    except subprocess.TimeoutExpired:
        generated_text = "Generation took too long and was terminated."
    except Exception as e:
        generated_text = f"An error occurred: {str(e)}"

    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True)