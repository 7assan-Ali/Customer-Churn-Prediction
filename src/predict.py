from pathlib import Path
import pandas as pd
from joblib import load

ROOT=Path(__file__).resolve().parents[1]
model=load(ROOT/"models/model.joblib")
sample=ROOT/"data/raw/sample.csv"
if not sample.exists(): raise FileNotFoundError("Create data/raw/sample.csv with the training feature columns.")
df=pd.read_csv(sample)
print(pd.DataFrame({"prediction":model.predict(df)}))
