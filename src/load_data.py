import pandas as pd
import os

# def prepare_data():
#     input_file = "/home/jano/Documentos/MIOTI/MLOps/Challenge/Bitcoin.csv"  
#     output_dir = "data/raw"
#     output_file = os.path.join(output_dir, "bitcoin_raw.csv")
# 
#     os.makedirs(output_dir, exist_ok=True)
# 
#     df = pd.read_csv(input_file)
# 
    # Aunque en el EDA he visto que no había nulos, por si acaso hago esto:
#     df_clean = df.dropna()
# 
#     df_clean.to_csv(output_file, index=False)

# if __name__ == "__main__":
#     prepare_data()

def prepare_data():
    input_file = "/home/jano/Documentos/MIOTI/MLOps/Challenge/Bitcoin.csv"  
    output_dir = "data/raw"
    output_file = os.path.join(output_dir, "bitcoin_raw.csv")

    os.makedirs(output_dir, exist_ok=True)

    print("Cargando dataset original")
    df = pd.read_csv(input_file)
    df_clean = df.dropna().copy()

    # Convertir el Timestamp (float64) a formato Fecha real para poder agrupar
    # (unit='s' porque está en Unix)
    df_clean['Date'] = pd.to_datetime(df_clean['Timestamp'], unit='s')
    df_clean.set_index('Date', inplace=True)

    print("Agrupando datos por día (calculando medias diarias)...")
    # Resampleo por día ('D'). Media de los precios y suma del volumen
    df_daily = df_clean.resample('D').agg({
        'Open': 'mean',
        'High': 'mean',
        'Low': 'mean',
        'Close': 'mean',
        'Volume': 'sum'
    }).dropna()

    # Columna 'Timestamp' otra vez en formato numérico
    # Así el script de train.py no se rompe al buscar la columna original.
    df_daily = df_daily.reset_index()
    df_daily['Timestamp'] = df_daily['Date'].astype('int64') // 10**9
    
    # Dejo las columnas igual que al principio
    df_final = df_daily[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]

    print(f"Dataset reducido con éxito. Nuevas dimensiones: {df_final.shape}")
    df_final.to_csv(output_file, index=False)

if __name__ == "__main__":
    prepare_data()
