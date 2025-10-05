# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 16:26:53 2025

@author: Admin
"""

import os
import numpy as np
import pandas as pd
import random
import time
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RationalQuadratic
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def reset_random_seeds():
    os.environ['PYTHONHASHSEED'] = str(1)
    np.random.seed(1)
    random.seed(1)

def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print("Model Evaluation:")
    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}\n")

def train_and_save_model(data_file="DATA324_Modeling.xlsx", model_file='gpr_model2.pkl', scaler_file='scaler2.pkl'):
    reset_random_seeds()
    df = pd.read_excel(data_file, engine='openpyxl')
    df.columns = df.columns.str.strip()

    # Define input/output
    feature_columns = [col for col in df.columns if col != 'CONCRETE']
    target_column = 'CONCRETE'

    X = df[feature_columns]
    y = df[target_column]

    # Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RationalQuadratic()
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1.0, random_state=0)
    gpr.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = gpr.predict(X_test_scaled)
    evaluate_model(y_test, y_pred)

    # Save model and scaler
    joblib.dump(gpr, model_file)
    joblib.dump(scaler, scaler_file)
    print(f"Model saved as '{model_file}', Scaler saved as '{scaler_file}'")

def predict_batch(input_df, model_file='gpr_model2.pkl', scaler_file='scaler2.pkl'):
    # Load model and scaler
    gpr = joblib.load(model_file)
    scaler = joblib.load(scaler_file)

    # Scale and predict
    X_scaled = scaler.transform(input_df)
    predictions = gpr.predict(X_scaled)
    input_df = input_df.copy()
    input_df["Predicted_CS"] = predictions
    return input_df

def main():
    start = time.time()
    
    # Step 1: Train and save the model
    train_and_save_model()

    # Step 2: Prepare new data for prediction
    
    #51-55 MPa
    data_dict = {
        'OPC':     [186.25	,
166.75	,
213.85	,
218.49	,
220.16	,
255.29	,
248.07	,
247.68	,
265.68	,
302.38	,
322.93	,
331.46	,
354.5	,
365.94	,
368.47	,
367.55	,
400.23	,
406.25	,
401.43	,
449.16	,
451.27	
],
        'BFS':     [0.1	,
50.97	,
0.1	,
7.1	,
29.95	,
0.1	,
32.35	,
47.85	,
38.49	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	
],
        'FLY ASH':  [0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	
],
        'WATER':   [121.75	,
121.75	,
121.75	,
121.75	,
121.75	,
121.75	,
121.75	,
132.21	,
121.75	,
135	,
134.53	,
125.79	,
132.64	,
124.83	,
143.45	,
136.92	,
126.77	,
140.11	,
133	,
148.45	,
139.99	
],
        'SP':      [0.1	,
0.1	,
0.1	,
1.06	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.1	,
0.41	,
0.1	,
0.1	,
1.16	,
0.1	,
0.1	
],
        'CA':      [1140.52	,
1145	,
1136.41	,
1133.66	,
1145	,
1145	,
1107.58	,
1145	,
1122.46	,
1114.9	,
1145	,
1133.76	,
1143.55	,
1120.74	,
1055.59	,
1063.26	,
1074.93	,
1086.39	,
1068.05	,
996.7	,
1013.97	
],
        'FA':      [848.35	,
806.8	,
828.59	,
818.37	,
787.16	,
788.68	,
796.38	,
725.44	,
762.48	,
746.7	,
702.79	,
728.49	,
682.79	,
716.45	,
727.79	,
739.49	,
727.23	,
679.2	,
714.41	,
706.64	,
713.74	
],
        'AGE':  [28]*21
    }

    df_input = pd.DataFrame(data_dict)

    # Step 3: Predict CS
    result_df = predict_batch(df_input)

    # Step 4: Show and optionally export
    print(result_df[["OPC", "BFS", "FLY ASH", "WATER", "SP", "CA", "FA", "AGE", "Predicted_CS"]])
    result_df.to_excel("Predicted_CS_Output.xlsx", index=False)

    print(f"\n✅ Completed in {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    main()

