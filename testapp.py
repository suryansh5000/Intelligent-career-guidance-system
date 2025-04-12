from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(_name_)

@app.route('/')
def home():
    return render_template("hometest.html")

@app.route('/predict', methods=['POST'])
def result():
    if request.method == 'POST':
        # Collect form data
        result = request.form
        res = result.to_dict(flat=True)

        try:
            # Convert all values to float
            arr = [float(value) for value in res.values()]
        except ValueError:
            return "⚠ Error: All form inputs must be numeric."

        # Reshape for prediction
        data = np.array(arr).reshape(1, -1)

        # Load model
        loaded_model = pickle.load(open("careerlast.pkl", 'rb'))

        # Predict main job
        predictions = loaded_model.predict(data)
        pred = loaded_model.predict_proba(data)
        pred = pred > 0.05

        # Build result mappings
        res_dict = {}
        final_res = {}
        index = 0
        for j in range(pred.shape[1]):
            if pred[0, j]:
                res_dict[index] = j
                index += 1

        index = 0
        for key, value in res_dict.items():
            if value != predictions[0]:
                final_res[index] = value
                index += 1

        # Job titles dictionary
        jobs_dict = {
            0: 'AI ML Specialist',
            1: 'API Integration Specialist',
            2: 'Application Support Engineer',
            3: 'Business Analyst',
            4: 'Customer Service Executive',
            5: 'Cyber Security Specialist',
            6: 'Data Scientist',
            7: 'Database Administrator',
            8: 'Graphics Designer',
            9: 'Hardware Engineer',
            10: 'Helpdesk Engineer',
            11: 'Information Security Specialist',
            12: 'Networking Engineer',
            13: 'Project Manager',
            14: 'Software Developer',
            15: 'Software Tester',
            16: 'Technical Writer'
        }

        job0 = predictions[0]
        return render_template("testafter.html", final_res=final_res, job_dict=jobs_dict, job0=job0)

if _name_ == '_main_':
    app.run(debug=True)
