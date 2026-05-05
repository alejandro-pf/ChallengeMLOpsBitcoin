import pandas as pd
import os

def prepare_data():
    input_file = "/home/jano/Documentos/MIOTI/MLOps/Challenge/Bitcoin.csv"  
    output_dir = "data/raw"
    output_file = os.path.join(output_dir, "bitcoin_raw.csv")

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_file)

    # Aunque en el EDA he visto que no había nulos, por si acaso hago esto:
    df_clean = df.dropna()

    df_clean.to_csv(output_file, index=False)

if __name__ == "__main__":
    prepare_data()
