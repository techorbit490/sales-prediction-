"""
Sales Prediction — Flask Backend API (Fixed)
"""
import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder

BASE   = os.path.dirname(os.path.abspath(__file__))
FRONT  = os.path.join(BASE, '..', 'frontend')

app = Flask(__name__,
            static_folder=os.path.join(FRONT, 'static'),
            template_folder=os.path.join(FRONT, 'templates'))
CORS(app)

# Load model (NO scaler needed for GradientBoosting)
model = joblib.load(os.path.join(BASE, 'best_model.pkl'))

# Rebuild label encoders from training data
df_train = pd.read_csv(os.path.join(BASE, 'sales_data.csv'))
ENCODERS = {}
for col in ['Platform', 'Customer_Segment', 'Campaign_Type', 'Region']:
    le = LabelEncoder()
    le.fit(df_train[col].astype(str))
    ENCODERS[col] = le

FEATURES = [
    'Log_Ad_Spend', 'Campaign_Score', 'Competitor_Index', 'Economic_Index',
    'Discount_Pct', 'Channels_Used', 'CSAT_Score', 'Log_Traffic',
    'Seasonal_Factor', 'Spend_per_Channel', 'Score_x_Season',
    'Month', 'Quarter',
    'Platform_Enc', 'Customer_Segment_Enc', 'Campaign_Type_Enc', 'Region_Enc'
]

@app.route('/')
def index():
    return send_from_directory(os.path.join(FRONT, 'templates'), 'index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'GradientBoostingRegressor'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        d = request.get_json(force=True)

        ad_spend    = float(d['ad_spend'])
        channels    = max(1, int(d['channels_used']))
        camp_score  = float(d['campaign_score'])
        seasonal    = float(d['seasonal_factor'])
        traffic     = float(d['website_traffic'])
        month       = int(d['month'])
        quarter     = (month - 1) // 3 + 1

        row = pd.DataFrame([{
            'Log_Ad_Spend'         : np.log1p(ad_spend),
            'Campaign_Score'       : camp_score,
            'Competitor_Index'     : float(d['competitor_index']),
            'Economic_Index'       : float(d['economic_index']),
            'Discount_Pct'         : float(d['discount_pct']),
            'Channels_Used'        : channels,
            'CSAT_Score'           : float(d['csat_score']),
            'Log_Traffic'          : np.log1p(traffic),
            'Seasonal_Factor'      : seasonal,
            'Spend_per_Channel'    : ad_spend / channels,
            'Score_x_Season'       : camp_score * seasonal,
            'Month'                : month,
            'Quarter'              : quarter,
            'Platform_Enc'         : int(ENCODERS['Platform'].transform([d['platform']])[0]),
            'Customer_Segment_Enc' : int(ENCODERS['Customer_Segment'].transform([d['customer_segment']])[0]),
            'Campaign_Type_Enc'    : int(ENCODERS['Campaign_Type'].transform([d['campaign_type']])[0]),
            'Region_Enc'           : int(ENCODERS['Region'].transform([d['region']])[0]),
        }])[FEATURES]

        prediction = float(model.predict(row)[0])
        prediction = max(0, round(prediction, 2))
        roi        = round(((prediction - ad_spend) / ad_spend) * 100, 1) if ad_spend > 0 else 0

        return jsonify({
            'success'        : True,
            'predicted_sales': prediction,
            'roi_pct'        : roi,
            'profit'         : round(prediction - ad_spend, 2)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    print("\n🚀 Sales Prediction API → http://localhost:5000\n")
    app.run(debug=True, port=5000)
