import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
import mysql.connector
from flask_cors import CORS

pickled_model = pickle.load(open('random_forest_model.pkl', 'rb'))

app = Flask(__name__, template_folder='templates')


#app.config['MONGO_DBNAME'] = 'patient_database'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://j6qbx3bgjysst4jr:mcbsdk2s27ldf37t@frwahxxknm9kwy6c.cbetxkdyhwsb.us-east-1.rds.amazonaws.com:3306/nkw2tiuvgv6ufu1z'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Optional, but recommended to suppress SQLAlchemy deprecation warnings
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
    #online_uses = mongo.db.users.find({"online": True})
    return "Thryoid Disease Detection"


@app.route('/predict', methods=['POST'])
def predict():

    #db = mongo.db.patient_data_collection
   
    data = request.get_json()
    cursor = mysql.cursor()

    # Define the SQL query with placeholders


    # Extract data values
    new_data = (
        data['age'], data['sex'], data['TSH'], data['T3'], data['T4U'], data['FTI'],
        data['onthyroxine'], data['queryonthyroxine'], data['onantithyroidmedication'],
        data['sick'], data['pregnant'], data['thyroidsurgery'], data['I131treatment'],
        data['queryhypothyroid'], data['queryhyperthyroid'], data['lithium'],
        data['goitre'], data['tumor'], data['hypopituitary'], data['psych']
    )



    df_transform = pd.DataFrame.from_dict([new_data])

  
    age = new_data[0]

    df_transform['age'] = age ** (1 / 2)
    print(df_transform.age)


    TSH = new_data[2]
    df_transform['TSH'] = np.log1p(TSH)
    # print(df_transform.TSH)
    T3=new_data[3]
    df_transform.T3 = T3 ** (1 / 2)
    # print(df_transform.T3)
    T4U=new_data[4]

    df_transform.T4U = np.log1p(T4U)
    # print(df_transform.T4U)
    FTI=new_data[5]
    df_transform.FTI = FTI ** (1 / 2)
    print(FTI)

    df_transform.to_dict()

    arr = np.array([[df_transform.age, new_data[1],
                     df_transform.TSH, df_transform.T3, df_transform.T4U, df_transform.FTI,
                     new_data[6], new_data[7],
                     new_data[8],
                     new_data[9], new_data[10], new_data[11],
                     new_data[12],
                     new_data[13], new_data[14],
                     new_data[15], new_data[16], new_data[17],
                     new_data[18],
                     new_data[19]]])

    # print("After transformation:\n")
    # print(arr)

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

    return jsonify(res_Val)

if __name__ == '__main__':
    app.run(debug=True)
