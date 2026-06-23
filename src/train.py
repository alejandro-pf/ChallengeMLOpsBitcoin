import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pickle

mlflow.set_experiment("Bitcoin_Price_Prediction")

def train_model(n_estimators=100, max_depth=10):

	
	# Limpio y configuro datos
        df = pd.read_csv("data/raw/bitcoin_raw.csv")
        
        
        # Quiero predecir el siguiente minuto en función de los datos de "este minuto" (ahora día), por lo tanto el target será el precio 
        # del siguiente minuto (ahora día), así el modelo entrena con los datos del minuto (ahora día) presente para predecir el siguiente.
        
        df['target'] = df['Close'].shift(-1)
        df = df.dropna()
        
        X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        y = df['target']
        
        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]



	# Entreno el modelo
        with mlflow.start_run(run_name=f"RandomForest_est_{n_estimators}_depth_{max_depth}"):
        
        
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
            model.fit(X_train, y_train)


            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)



            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
        

            mlflow.sklearn.log_model(model, "model")
        
            print(f"Modelo entrenado - RMSE: {rmse}")

        
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)

if __name__ == "__main__":

    train_model(n_estimators=100, max_depth=10)
