import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

np.random.seed(42)

try:
    os.makedirs("datasets")
except:
    pass

print("generating 50k rows...")

cities=["Helsinki","Espoo","Tampere","Turku","Oulu"]
networks=["Fiber","5G","LTE","DSL"]
severities=["Critical","High","Medium","Low"]
tiers=["Basic","Premium","Enterprise"]
channels=["Email","Phone","Monitoring","Portal"]
causes=["Hardware","Software","Config","Weather","Unknown"]

nrows=50000
data=[]

# dates
all_dt=pd.date_range(start="2023-01-01",end="2025-12-31 23:00",freq="H")
picked=np.random.choice(all_dt,nrows,replace=True)

for i in range(nrows):
    rid=1000+i
    
    city=np.random.choice(cities,p=[0.3,0.2,0.2,0.1,0.2])
    net=np.random.choice(networks)
    sev=np.random.choice(severities,p=[0.1,0.3,0.4,0.2])
    tier=np.random.choice(tiers,p=[0.5,0.3,0.2])
    cause=np.random.choice(causes)
    ch=np.random.choice(channels,p=[0.3,0.2,0.3,0.2])
    reopened=np.random.poisson(0.25)
    
    dt=pd.to_datetime(picked[i])
    opened_date=dt.strftime("%Y-%m-%d")
    opened_month=dt.strftime("%Y-%m")
    
    # sla logic
    if sev=="Critical":
        target=60
        base=50
    elif sev=="High":
        target=240
        base=180
    elif sev=="Medium":
        target=480
        base=400
    else:
        target=1440
        base=1000
        
    noise=np.random.normal(0,target*0.35)
    
    # biases
    if city in ["Oulu","Turku"]: noise+=25
    if city=="Helsinki": noise-=10
    if cause=="Hardware": noise+=20
    if cause=="Config": noise-=10
    if tier=="Enterprise": noise-=20
    if ch=="Monitoring": noise-=10
    noise+=reopened*35
    
    res=max(5,base+noise)
    res=round(res,1)
    
    breach=1 if res>target else 0
    
    cost=0
    if breach==1:
        if tier=="Enterprise": cost=5000
        elif tier=="Premium": cost=1000
        else: cost=200
    
    rec={"incident_id":rid,"opened_date":opened_date,"opened_month":opened_month,"site_city":city,"network_type":net,"severity":sev,"customer_tier":tier,"ticket_channel":ch,"cause_category":cause,"reopened_count":reopened,"sla_target_minutes":target,"resolution_minutes":res,"sla_breached":breach,"penalty_cost":cost}
    data.append(rec)

df=pd.DataFrame(data)
print("training...")

# prep
df_ml=pd.get_dummies(df,columns=["site_city","network_type","severity","customer_tier","ticket_channel","cause_category"],drop_first=True)

X=df_ml.drop(columns=["incident_id","opened_date","opened_month","resolution_minutes","sla_breached","sla_target_minutes","penalty_cost"])
y=df_ml["sla_breached"]
X=X.fillna(0)

# added stratify to keep ratio
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

lr=LogisticRegression(class_weight="balanced",max_iter=1000,random_state=42)
lr.fit(X_train,y_train)

rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train,y_train)

# predictions
p_lr=lr.predict_proba(X)[:,1]
p_rf=rf.predict_proba(X)[:,1]

df["lr_prob"]=np.round(p_lr,4)
df["rf_prob"]=np.round(p_rf,4)
df["risk_score"]=df[["lr_prob","rf_prob"]].max(axis=1)

conds=[(df["risk_score"]<0.4), (df["risk_score"]>=0.4)&(df["risk_score"]<0.7), (df["risk_score"]>=0.7)]
df["risk_band"]=np.select(conds,["Low","Medium","High"],default="Unknown")
df["breach_flag"]=df["sla_breached"]
df["incident_count_hint"]=1

# metrics
yp_lr=lr.predict(X_test)
df["metric_lr_acc"]=accuracy_score(y_test,yp_lr)
df["metric_lr_prec"]=precision_score(y_test,yp_lr,zero_division=0)
df["metric_lr_rec"]=recall_score(y_test,yp_lr,zero_division=0)

yp_rf=rf.predict(X_test)
df["metric_rf_acc"]=accuracy_score(y_test,yp_rf)
df["metric_rf_prec"]=precision_score(y_test,yp_rf,zero_division=0)
df["metric_rf_rec"]=recall_score(y_test,yp_rf,zero_division=0)

# drivers for powerbi
cols=list(X.columns)
c=lr.coef_[0]
abs_c=np.abs(c)
top_idx=np.argsort(abs_c)[::-1][:6]

for idx in top_idx:
    fname=cols[idx]
    val=c[idx]
    df["imp_"+fname]=round(val,4)

df.to_csv("datasets/dashboard_data.csv",index=False)
print("saved dashboard_data.csv")
print("done")