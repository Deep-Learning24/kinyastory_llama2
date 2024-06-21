from flask import Flask, request, jsonify, render_template
import subprocess
import nltk
from nltk.translate.bleu_score import sentence_bleu
import math

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
        '-m', mode
    ]

    if system_prompt:
        cmd.extend(['-y', system_prompt])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout.strip()

        # Calculate perplexity
        tokens = nltk.word_tokenize(output)  # Tokenize the output text
        perplexity = calculate_perplexity(tokens)
        
        # Calculate BLEU score
        reference = prompt.split()  # Assuming prompt is the reference for BLEU score
        candidate = output.split()
        bleu_score = sentence_bleu([reference], candidate)

        return jsonify({'output': output, 'perplexity': perplexity, 'bleu_score': bleu_score})
    except subprocess.CalledProcessError as e:
        return jsonify({'error': str(e), 'output': e.output}), 400
    
def calculate_perplexity(tokens):
    token_tuples = list(nltk.ngrams(tokens, 2))  # Convert tokens to bigrams (tuples)
    bigram_model = nltk.lm.models.KneserNeyInterpolated(2)
    bigram_model.fit([token_tuples], vocabulary_text=nltk.lm.Vocabulary(tokens))
    return math.exp(bigram_model.perplexity(token_tuples))

if __name__ == '__main__':
    app.run(debug=True)
