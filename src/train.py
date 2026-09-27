from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,RandomForestRegressor,GradientBoostingRegressor

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/raw/dataset.csv"; MODEL_DIR=ROOT/"models"; MODEL_DIR.mkdir(exist_ok=True)
if not DATA.exists():
    DATA.parent.mkdir(parents=True,exist_ok=True)
    import urllib.request
    urllib.request.urlretrieve("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",DATA)

df=pd.read_csv(DATA)
target="Churn"
if target not in df.columns:
    match=[c for c in df.columns if c.lower()==target.lower()]
    if match: target=match[0]
    else: raise ValueError(f"Target not found: {df.columns.tolist()}")
X=df.drop(columns=[target]); y=df[target]

X=X.drop(columns=[c for c in X.columns if c.lower() in {"id","customerid","customer_id"}],errors="ignore")
num=X.select_dtypes(include=np.number).columns.tolist(); cat=X.select_dtypes(exclude=np.number).columns.tolist()
pre=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),num),("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)])
Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
models={"Logistic Regression":LogisticRegression(max_iter=2000),"Random Forest":RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1,class_weight="balanced"),"Gradient Boosting":GradientBoostingClassifier(random_state=42)}
for name,est in models.items():
    pipe=Pipeline([("preprocess",pre),("model",est)]); pipe.fit(Xt,yt); pred=pipe.predict(Xv)
    row={"accuracy":accuracy_score(yv,pred),"precision":precision_score(yv,pred,zero_division=0),"recall":recall_score(yv,pred,zero_division=0),"f1":f1_score(yv,pred,zero_division=0)}
    if hasattr(pipe,"predict_proba"): row["roc_auc"]=roc_auc_score(yv,pipe.predict_proba(Xv)[:,1])
    print(name,row)
final_name=list(models)[-1]
final=Pipeline([("preprocess",pre),("model",models[final_name])]); final.fit(Xt,yt)
dump(final,MODEL_DIR/"model.joblib"); print("Saved",final_name,"to",MODEL_DIR/"model.joblib")
