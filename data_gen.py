import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score

np.random.seed(42)

out_dir="datasets"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

n_rows=10000
ids=np.arange(1000,1000+n_rows)
cities=["Helsinki","Espoo","Tampere","Turku","Oulu"]
network_types=["Fiber","5G","LTE","DSL"]
severities=["Critical","High","Medium","Low"]
channels=["Email","Phone","Monitoring","Portal"]
causes=["Hardware","Software","Config","Unknown"]
tiers=["Basic","Premium","Enterprise"]

# random looking dates (strings)
dt=pd.date_range(start="2023-01-01",end="2025-12-31 23:00",freq="H")
dt=np.random.choice(dt,n_rows,replace=True)
opened_date=[pd.to_datetime(x).strftime("%Y-%m-%d") for x in dt]
opened_month=[pd.to_datetime(x).strftime("%Y-%m") for x in dt]

city_col=np.random.choice(cities,n_rows,p=[0.35,0.18,0.18,0.12,0.17])
net_col=np.random.choice(network_types,n_rows)
sev_col=np.random.choice(severities,n_rows,p=[0.1,0.3,0.4,0.2])
chan_col=np.random.choice(channels,n_rows)
cause_col=np.random.choice(causes,n_rows)
tier_col=np.random.choice(tiers,n_rows)
reopened=np.random.poisson(0.25,n_rows)

sla_targets=[]
resolution_mins=[]
breached=[]

# resolution time logic
for i in range(n_rows):
    sev=sev_col[i]
    city=city_col[i]
    cause=cause_col[i]
    tier=tier_col[i]
    ch=chan_col[i]
    ro=reopened[i]

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

    # city stuff
    if city in ["Oulu","Turku"]:
        noise +=25
    if city=="Helsinki":
        noise -=10

    # cause stuff
    if cause=="Hardware":
        noise +=20
    elif cause=="Config":
        noise -=10

    # tier
    if tier=="Enterprise":
        noise -=20
    elif tier=="Basic":
        noise +=15

    # channel
    if ch=="Monitoring":
        noise -=10

    # reopened count pushes it up
    noise += ro*35

    final_res=max(5,base+noise)

    sla_targets.append(target)
    resolution_mins.append(round(final_res,1))
    breached.append(1 if final_res>target else 0)

df_incidents=pd.DataFrame({"incident_id": ids,"opened_date": opened_date,"opened_month": opened_month,"site_city": city_col,
    "network_type": net_col,"severity": sev_col,"sla_target_minutes": sla_targets,"resolution_minutes": resolution_mins,"sla_breached": breached,"ticket_channel": chan_col,
    "cause_category": cause_col,"reopened_count": reopened, "customer_tier": tier_col})

print("incidents made:",len(df_incidents))
print("breach rate is", f"{df_incidents['sla_breached'].mean():.2%}")

df_ml=df_incidents.copy()

cat_cols=["site_city","network_type","severity","ticket_channel","cause_category","customer_tier"]
df_ml=pd.get_dummies(df_ml,columns=cat_cols,drop_first=True)

drop_cols=["incident_id","opened_date","opened_month","resolution_minutes","sla_breached","sla_target_minutes"]
X=df_ml.drop(columns=[c for c in drop_cols if c in df_ml.columns])
y=df_ml["sla_breached"]

X=X.fillna(0)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

lr=LogisticRegression(class_weight="balanced",max_iter=1000,random_state=42)
lr.fit(X_train,y_train)

rf=RandomForestClassifier(n_estimators=200,random_state=42)
rf.fit(X_train,y_train)

all_probs_lr=lr.predict_proba(X)[:,1]
all_probs_rf=rf.predict_proba(X)[:,1]
all_preds_lr=(all_probs_lr>=0.5).astype(int)
all_preds_rf=(all_probs_rf>=0.5).astype(int)

split_tag=np.array(["train"]*len(X))
split_tag[X_test.index.values]="test"

df_preds=pd.DataFrame({"incident_id": df_incidents["incident_id"],"lr_prob": np.round(all_probs_lr,4),"rf_prob": np.round(all_probs_rf,4),"lr_pred": all_preds_lr,"rf_pred": all_preds_rf,"data_split": split_tag})

agreement=[]
more_risky=[]
diffs=[]

for i in range(len(df_preds)):
    if df_preds.loc[i,"lr_pred"]==df_preds.loc[i,"rf_pred"]:
        agreement.append("Agree")
    else:
        agreement.append("Disagree")

    p1=df_preds.loc[i,"lr_prob"]
    p2=df_preds.loc[i,"rf_prob"]
    if p1>p2:
        more_risky.append("LR")
    elif p2>p1:
        more_risky.append("RF")
    else:
        more_risky.append("Equal")

    diffs.append(abs(p1-p2))

df_preds["model_agreement"]=agreement
df_preds["model_more_risky"]=more_risky
df_preds["prob_diff"]=np.round(diffs,4)

y_pred_lr=lr.predict(X_test)
y_prob_lr=lr.predict_proba(X_test)[:,1]
y_pred_rf=rf.predict(X_test)
y_prob_rf=rf.predict_proba(X_test)[:,1]

cm_lr=confusion_matrix(y_test,y_pred_lr,labels=[0,1])
tn_lr,fp_lr,fn_lr,tp_lr=cm_lr.ravel()

cm_rf=confusion_matrix(y_test,y_pred_rf,labels=[0,1])
tn_rf,fp_rf,fn_rf,tp_rf=cm_rf.ravel()

metrics_lr={"accuracy": accuracy_score(y_test,y_pred_lr),"precision": precision_score(y_test,y_pred_lr,zero_division=0),"recall": recall_score(y_test,y_pred_lr,zero_division=0),"f1": f1_score(y_test,y_pred_lr,zero_division=0),"roc_auc": roc_auc_score(y_test,y_prob_lr),"avg_precision": average_precision_score(y_test,y_prob_lr), "tp": tp_lr,"fp": fp_lr,"tn": tn_lr,"fn": fn_lr}

metrics_rf={"accuracy": accuracy_score(y_test,y_pred_rf),"precision": precision_score(y_test,y_pred_rf,zero_division=0),"recall": recall_score(y_test,y_pred_rf,zero_division=0),"f1": f1_score(y_test,y_pred_rf,zero_division=0),
            "roc_auc": roc_auc_score(y_test,y_prob_rf),"avg_precision": average_precision_score(y_test,y_prob_rf),"tp": tp_rf,"fp": fp_rf,"tn": tn_rf,"fn": fn_rf}

thresholds=[0.3,0.5,0.7]

lr_thresh_data={}
for t in thresholds:
    preds=(y_prob_lr>=t).astype(int)
    t_str=str(t).replace(".","")
    lr_thresh_data["lr_precision_t"+t_str]=precision_score(y_test,preds,zero_division=0)
    lr_thresh_data["lr_recall_t"+t_str]=recall_score(y_test,preds,zero_division=0)
    lr_thresh_data["lr_f1_t"+t_str]=f1_score(y_test,preds,zero_division=0)

rf_thresh_data={}
for t in thresholds:
    preds=(y_prob_rf>=t).astype(int)
    t_str=str(t).replace(".","")
    rf_thresh_data["rf_precision_t"+t_str]=precision_score(y_test,preds,zero_division=0)
    rf_thresh_data["rf_recall_t"+t_str]=recall_score(y_test,preds,zero_division=0)
    rf_thresh_data["rf_f1_t"+t_str]=f1_score(y_test,preds,zero_division=0)

df_metrics=pd.DataFrame([{"model_name":"LogisticRegression",**metrics_lr},{"model_name":"RandomForest",**metrics_rf}])

df_dash=df_incidents.copy()
df_dash=df_dash.merge(df_preds,on="incident_id",how="left")

df_dash["risk_score"]=df_dash[["lr_prob","rf_prob"]].max(axis=1)
df_dash["risk_band"]=np.select([(df_dash["risk_score"]<0.4),(df_dash["risk_score"]>=0.4) & (df_dash["risk_score"]<0.7),
        (df_dash["risk_score"]>=0.7)],["Low","Medium","High"],default="Unknown")

df_dash["breach_flag"]=df_dash["sla_breached"]
df_dash["breach_rate_hint"]=df_dash["sla_breached"]
df_dash["incident_count_hint"]=1

# quick feature importance strings (for me)
feat_names=list(X.columns)

lr_coefs=lr.coef_[0]
lr_abs=np.abs(lr_coefs)
lr_top_idx=np.argsort(lr_abs)[::-1][:10]
lr_top_text=[]
for idx in lr_top_idx:
    lr_top_text.append(feat_names[idx]+":"+str(round(lr_coefs[idx],4)))
df_dash["lr_top_features"]=";".join(lr_top_text)

rf_imp=rf.feature_importances_
rf_top_idx=np.argsort(rf_imp)[::-1][:10]
rf_top_text=[]
for idx in rf_top_idx:
    rf_top_text.append(feat_names[idx]+":"+str(round(rf_imp[idx],4)))
df_dash["rf_top_features"]=";".join(rf_top_text)

for k,v in metrics_lr.items():
    df_dash["lr_"+k]=v
for k,v in metrics_rf.items():
    df_dash["rf_"+k]=v
for k,v in lr_thresh_data.items():
    df_dash[k]=v
for k,v in rf_thresh_data.items():
    df_dash[k]=v

df_incidents.to_csv(os.path.join(out_dir,"incidents.csv"),index=False)
df_preds.to_csv(os.path.join(out_dir,"predictions.csv"),index=False)
df_metrics.to_csv(os.path.join(out_dir,"model_metrics.csv"),index=False)
df_dash.to_csv(os.path.join(out_dir,"dashboard_data.csv"),index=False)

print("done")