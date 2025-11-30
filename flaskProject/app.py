from flask import Flask, render_template, request, jsonify
import nbformat

app = Flask(__name__)

# Funksjon for å laste og kjøre Jupyter-notatboken
def load_and_run_notebook(notebook_path):
    with open(notebook_path, "r") as file:
        notebook = nbformat.read(file, as_version=4)

    exec_context = {}

    # kjører hver kodecelle i jupter nottebook, og filtrer ut magiske kommandoer som mamtplotlib osv
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            code_lines = cell.source.split('\n')
            filtered_lines = [line for line in code_lines if not line.strip().startswith('%')]
            filtered_code = '\n'.join(filtered_lines)
            exec(filtered_code, exec_context)

    return exec_context

notebook_path = "C:/Users/ayebe/AI-Arch/fortuna_plain.ipynb"
exec_context = load_and_run_notebook(notebook_path)

# Hent  funksjon og parametere fra fortuna
model_function = exec_context.get('f')  # funksjonen f(x, theta)
best_theta = exec_context.get('best_theta')  # De beste parametere funnet

# Definer en wrapper-funksjon for prediksjon
def predict_time_for_height(height):
    try:
        # Hvis modelfunksjonen predikerer tid gitt høyde, sørg for dette forholdet
        time = model_function(height, best_theta)
        print(f"Forventet tid for høyde {height}: {time}")  # Debugging output
        return time
    except Exception as e:
        print(f"Feil i Jupyter-beregning: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Antar at input faktisk er høyde hvis det var ment å være tid
        height = float(request.form['time'])  # Faktisk passerer høyde i stedet
        time = predict_time_for_height(height)  # Bruk den justerte funksjonen
        if time is None:
            return jsonify({'error': 'Beregning feilet.'}), 400
        return jsonify({'height': time})  # Send JSON-respons med forventet tid
    except Exception as e:
        print(f"Feil under prediksjon: {e}")
        return jsonify({'error': 'En feil oppstod under prediksjonen.'}), 400

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # Vis skjemaet først

if __name__ == '__main__':
    app.run(debug=True)
