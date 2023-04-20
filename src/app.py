from flask import Flask, request
from flask_cors import CORS
import pandas as pd
import pickle
import string
  
app = Flask(__name__)
CORS(app)

with open(r'C:\Users\INTEL\OneDrive\Desktop\Chatbot\src\Model\model.pkl', "rb") as f:
    model = pickle.load(f)
    
symptoms_descr = pd.read_csv(r'C:\Users\INTEL\OneDrive\Desktop\Chatbot\Dataset\symptom_Description.csv')
precaution_df = pd.read_csv('Dataset\symptom_precaution.csv')

@app.route('/',methods=['POST'])
def index():
    try:
        columns = ['itching', 'skin_rash', 'continuous_sneezing', 'shivering',
                   'stomach_pain', 'acidity', 'vomiting', 'indigestion',
                   'muscle_wasting', 'patches_in_throat', 'fatigue', 'weight_loss',
                   'sunken_eyes', 'cough', 'headache', 'chest_pain', 'back_pain',
                   'weakness_in_limbs', 'chills', 'joint_pain', 'yellowish_skin',
                   'constipation', 'pain_during_bowel_movements', 'breathlessness',
                   'cramps', 'weight_gain', 'mood_swings', 'neck_pain',
                   'muscle_weakness', 'stiff_neck', 'pus_filled_pimples',
                   'burning_micturition', 'bladder_discomfort', 'high_fever']
        
        keywords = {}
        for sym in columns:
            if '_' in sym:
                val = sym.split('_')
                keywords[sym] = val 
            else:
                keywords[sym] = [sym]
        
        data = {}
        input_syms = request.json['symps']
        input_syms = input_syms.translate(str.maketrans("", "", string.punctuation))
        input_syms = input_syms.lower().split(' ')
        for sym in input_syms:
            for key, val in keywords.items():
                if sym in val:
                    data[key] = 1
                else:
                    if key not in data:
                        data[key] = 0
        
        data = pd.DataFrame(data, index=[0])
        prediction = model.predict(data)
        desc = symptoms_descr[symptoms_descr['Disease'] == prediction[0]]['Description']
        precaution1 = precaution_df[precaution_df['Disease'] ==prediction[0]]['Precaution_1'].to_list()
        precaution2 = precaution_df[precaution_df['Disease'] ==prediction[0]]['Precaution_2'].to_list()
        precaution3 = precaution_df[precaution_df['Disease'] ==prediction[0]]['Precaution_3'].to_list()
        precaution4 = precaution_df[precaution_df['Disease'] ==prediction[0]]['Precaution_4'].to_list()
        desc = desc.to_list()
        return str(prediction[0])+' Description: '+str(desc[0]) + ' Precautions: ' + str(precaution1[0]) + ', ' + str(precaution2[0]) + ', ' + str(str(precaution3[0])) + ', '+ str(precaution4[0])
    
    except Exception as e:
        return str(e)

app.run(debug=True)