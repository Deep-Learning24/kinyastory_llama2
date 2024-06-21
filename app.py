from flask import Flask, request, jsonify, render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    mode = data.get('mode', 'generate')
    previous_output = data.get('previous_output', '')

    model_path = data.get('model_path', 'out/model.bin')
    tokenizer_path = data.get('tokenizer_path', 'data/tok2048.bin')
    temperature = data.get('temperature', 1.0)
    topp = data.get('topp', 0.9)
    steps = data.get('steps', 256)
    rng_seed = data.get('rng_seed', 0)
    system_prompt = data.get('system_prompt', '')
    check_performance = 1

    if mode == 'chat':
        if previous_output:
            prompt = previous_output + "\nUser: " + prompt
        else:
            prompt = "User: " + prompt

    # Construct the command
    cmd = [
        './run', model_path,
        '-i', prompt,
        '-z', tokenizer_path,
        '-t', str(temperature),
        '-p', str(topp),
        '-n', str(steps),
        '-s', str(rng_seed),
        '-m', mode,
    ]

    if system_prompt:
        cmd.extend(['-y', system_prompt])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        
        # #run evaluation
        eval_cmd = [
            './run', model_path,
        '-i', prompt,
        '-z', tokenizer_path,
        '-t', str(temperature),
        '-p', str(topp),
        '-n', str(steps),
        '-s', str(rng_seed),
        '-m', mode,
        '-g',output,
        '-c',str(check_performance)
        ]
        eval_result = subprocess.run(eval_cmd, capture_output=True, text=True, check=True)
        eval_output = eval_result.stdout.strip()
       
        return jsonify({'output': output,'eval_result':eval_output})
    except subprocess.CalledProcessError as e:
        return jsonify({'error': str(e), 'output': e.output}), 400
    


if __name__ == '__main__':
    app.run(debug=True)
