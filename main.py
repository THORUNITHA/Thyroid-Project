import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_cors import CORS
import mysql.connector

pickled_model = pickle.load(open('random_forest_model.pkl', 'rb'))

app = Flask(__name__)

CORS(app)

mysql_config = {
    'host': "frwahxxknm9kwy6c.cbetxkdyhwsb.us-east-1.rds.amazonaws.com",
    'user': "j6qbx3bgjysst4jr",
    'password': "mcbsdk2s27ldf37t",
    'database': "nkw2tiuvgv6ufu1z",
    'port': 3306,
}
mysql = mysql.connector.connect(**mysql_config)



@app.route('/')
def home():
    return render_template('index.html')

def predict_thyroid_disease(data):
    # Convert data to the appropriate data types

    data = request.get_json()
    cursor = mysql.cursor()


    age = data['age']
    sex = data['sex']
    TSH = data['TSH']
    T3 = data['T3']
    T4U = data['T4U']
    FTI = data['FTI']
    onthyroxine = data['onthyroxine']
    queryonthyroxine = data['queryonthyroxine']
    onantithyroidmedication = data['onantithyroidmedication']
    sick = data['sick']
    pregnant = data['pregnant']
    thyroidsurgery = data['thyroidsurgery']
    I131treatment = data['I131treatment']
    queryhypothyroid = data['queryhypothyroid']
    queryhyperthyroid = data['queryhyperthyroid']
    lithium = data['lithium']
    goitre = data['goitre']
    tumor = data['tumor']
    hypopituitary = data['hypopituitary']
    psych = data['psych']

    values = {
        "age": age, "sex": sex,
        "TSH": TSH, "T3": T3, "T4U": T4U, "FTI": FTI,
        "onthyroxine": onthyroxine, "queryonthyroxine": queryonthyroxine,
        "onantithyroidmedication": onantithyroidmedication,
        "sick": sick, "pregnant": pregnant, "thyroidsurgery": thyroidsurgery,
        "I131treatment": I131treatment,
        "queryhypothyroid": queryhypothyroid, "queryhyperthyroid": queryhyperthyroid,
        "lithium": lithium, "goitre": goitre, "tumor": tumor,
        "hypopituitary": hypopituitary,
        "psych": psych
    }

    df_transform = pd.DataFrame.from_dict([values])

    df_transform['age'] = pd.to_numeric(df_transform['age'], errors='coerce')
    df_transform['age'] = df_transform['age'] ** (1 / 2)
    df_transform['TSH'] = pd.to_numeric(df_transform['TSH'], errors='coerce')
    df_transform['TSH'] = np.log1p(df_transform['TSH'])
    df_transform['T3'] = pd.to_numeric(df_transform['T3'], errors='coerce')
    df_transform['T3'] = df_transform['T3'] ** (1 / 2)
    df_transform['T4U'] = pd.to_numeric(df_transform['T4U'], errors='coerce')
    df_transform['T4U'] = np.log1p(df_transform['T4U'])
    df_transform['FTI'] = pd.to_numeric(df_transform['FTI'], errors='coerce')
    df_transform['FTI'] = df_transform['FTI'] ** (1 / 2)

    arr = np.array([[df_transform['age'].values[0], sex,
                     df_transform['TSH'].values[0], df_transform['T3'].values[0], df_transform['T4U'].values[0], df_transform['FTI'].values[0],
                     onthyroxine, queryonthyroxine, onantithyroidmedication, sick, pregnant, thyroidsurgery,
                     I131treatment, queryhypothyroid, queryhyperthyroid, lithium, goitre, tumor, hypopituitary, psych]])

    pred = pickled_model.predict(arr)[0]

    if pred == 0:
        res_Val = "Hyperthyroid"
    elif pred == 1:
        res_Val = "Hypothyroid"
    else:
        res_Val = 'Negative'

    insert_query = """
    INSERT INTO thyroid_data 
    (age, sex, TSH, T3, T4U, FTI, onthyroxine, queryonthyroxine, onantithyroidmedication, 
    sick, pregnant, thyroidsurgery, I131treatment, queryhypothyroid, queryhyperthyroid, 
    lithium, goitre, tumor, hypopituitary, psych, result) 
    VALUES 
    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    new_data = (
        data['age'], data['sex'], data['TSH'], data['T3'], data['T4U'], data['FTI'],
        data['onthyroxine'], data['queryonthyroxine'], data['onantithyroidmedication'],
        data['sick'], data['pregnant'], data['thyroidsurgery'], data['I131treatment'],
        data['queryhypothyroid'], data['queryhyperthyroid'], data['lithium'],
        data['goitre'], data['tumor'], data['hypopituitary'], data['psych'], res_Val
    )

    cursor.execute(insert_query, new_data)
    mysql.commit()
    cursor.close()

    return res_Val

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    result = predict_thyroid_disease(data)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
