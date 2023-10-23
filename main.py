import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from flask_pymongo import PyMongo

import warnings
warnings.filterwarnings('ignore')

pickled_model = pickle.load(open('random_forest_model.pkl', 'rb'))

app = Flask(__name__)

app.config['MONGO_DBNAME'] = 'patient_database'
app.config[
    "MONGO_URI"] = 'mongodb://jananiim21:1234@ac-wqwdffe-shard-00-00.ckrfkgt.mongodb.net:27017,ac-wqwdffe-shard-00-01.ckrfkgt.mongodb.net:27017,ac-wqwdffe-shard-00-02.ckrfkgt.mongodb.net:27017/patient_database?ssl=true&replicaSet=atlas-vv8wvd-shard-0&authSource=admin&retryWrites=true&w=majority'

mongo = PyMongo(app)  # connector


@app.route('/')
def home():
    online_uses = mongo.db.users.find({"online": True})
    return jsonify(str(online_uses))


@app.route('/predict', methods=['POST'])
def predict():

    db = mongo.db.patient_data_collection

    data = request.get_json()

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

    # values = ({"age": [age], "sex": [sex],
    #            "TSH": TSH, "T3": T3, "T4U": T4U, "FTI": FTI,
    #            "onthyroxine": [onthyroxine], "queryonthyroxine": [queryonthyroxine],
    #            "onantithyroidmedication": [onantithyroidmedication],
    #            "sick": [sick], "pregnant": [pregnant], "thyroidsurgery": [thyroidsurgery],
    #            "I131treatment": [I131treatment],
    #            "queryhypothyroid": [queryhypothyroid], "queryhyperthyroid": [queryhyperthyroid],
    #            "lithium": [lithium], "goitre": [goitre], "tumor": [tumor],
    #            "hypopituitary": [hypopituitary],
    #            "psych": [psych]})

    values = ({"age": age, "sex": sex,
               "TSH": TSH, "T3": T3, "T4U": T4U, "FTI": FTI,
               "onthyroxine": onthyroxine, "queryonthyroxine": queryonthyroxine,
               "onantithyroidmedication": onantithyroidmedication,
               "sick": sick, "pregnant": pregnant, "thyroidsurgery": thyroidsurgery,
               "I131treatment": I131treatment,
               "queryhypothyroid": queryhypothyroid, "queryhyperthyroid": queryhyperthyroid,
               "lithium": lithium, "goitre": goitre, "tumor": tumor,
               "hypopituitary": hypopituitary,
               "psych": psych})

    insert_data = db.insert_one(values)

    df_transform = pd.DataFrame.from_dict([values])

    # print("applying transformation\n")

    df_transform.age = df_transform['age'] ** (1 / 2)
    print(df_transform.age)

    df_transform.TSH = np.log1p(df_transform['TSH'])
    # print(df_transform.TSH)
    #
    df_transform.T3 = df_transform['T3'] ** (1 / 2)
    # print(df_transform.T3)

    df_transform.T4U = np.log1p(df_transform['T4U'])
    # print(df_transform.T4U)

    df_transform.FTI = df_transform['FTI'] ** (1 / 2)
    # print(df_transform.FTI)

    df_transform.to_dict()

    arr = np.array([[df_transform.age, sex,
                     df_transform.TSH, df_transform.T3, df_transform.T4U, df_transform.FTI,
                     onthyroxine, queryonthyroxine,
                     onantithyroidmedication,
                     sick, pregnant, thyroidsurgery,
                     I131treatment,
                     queryhypothyroid, queryhyperthyroid,
                     lithium, goitre, tumor,
                     hypopituitary,
                     psych]])

    # print("After transformation:\n")
    # print(arr)

    pred = pickled_model.predict(arr)[0]

    if pred == 0:
        res_Val = "Hyperthyroid"
    elif pred == 1:
        res_Val = "Hypothyroid"
    else:
        res_Val = 'Negative'

    return jsonify(res_Val)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
