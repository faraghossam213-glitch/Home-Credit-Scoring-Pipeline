"""Reproducible Home Credit competition baseline + CV ensemble.
Run: python competition_model.py --data-dir ./input --output ./output
"""
from pathlib import Path
import argparse, json, numpy as np, pandas as pd
import lightgbm as lgb

SEED=42

def auc(y, p):
    y=np.asarray(y,dtype=np.int8); p=np.asarray(p,dtype=np.float64)
    ranks=pd.Series(p).rank(method="average").to_numpy()
    n1=y.sum(); n0=len(y)-n1
    return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

def features(df):
    x=df.drop(columns=[c for c in ["SK_ID_CURR","TARGET"] if c in df.columns]).copy()
    # Business-relevant, reproducible ratios; protect divisions by zero.
    x["CREDIT_TO_INCOME"]=x["AMT_CREDIT"]/(x["AMT_INCOME_TOTAL"].replace(0,np.nan))
    x["ANNUITY_TO_INCOME"]=x["AMT_ANNUITY"]/(x["AMT_INCOME_TOTAL"].replace(0,np.nan))
    x["CREDIT_TO_ANNUITY"]=x["AMT_CREDIT"]/(x["AMT_ANNUITY"].replace(0,np.nan))
    x["GOODS_TO_CREDIT"]=x["AMT_GOODS_PRICE"]/(x["AMT_CREDIT"].replace(0,np.nan))
    x["AGE_YEARS"]=-x["DAYS_BIRTH"]/365.25
    x["EMPLOYED_YEARS"]=x["DAYS_EMPLOYED"].clip(upper=0).abs()/365.25
    x["EMPLOYMENT_AGE_RATIO"]=x["EMPLOYED_YEARS"]/(x["AGE_YEARS"].replace(0,np.nan))
    x["INCOME_PER_FAMILY_MEMBER"]=x["AMT_INCOME_TOTAL"]/(x["CNT_FAM_MEMBERS"].replace(0,np.nan))
    src=[c for c in ["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"] if c in x]
    x["EXT_SOURCE_MEAN"]=x[src].mean(axis=1); x["EXT_SOURCE_MIN"]=x[src].min(axis=1)
    x["EXT_SOURCE_MAX"]=x[src].max(axis=1)
    if "EXT_SOURCE_2" in x and "EXT_SOURCE_3" in x:
        x["EXT_SOURCE_2_X_3"]=x["EXT_SOURCE_2"]*x["EXT_SOURCE_3"]
    return x

def stratified_folds(y, n_splits=5, seed=SEED):
    rng=np.random.default_rng(seed); folds=np.empty(len(y),dtype=np.int16)
    for cls in [0,1]:
        ix=np.flatnonzero(np.asarray(y)==cls); rng.shuffle(ix); folds[ix]=np.arange(len(ix))%n_splits
    return folds

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default='input'); ap.add_argument('--output',default='output'); ap.add_argument('--folds',type=int,default=5); args=ap.parse_args()
    d=Path(args.data_dir); o=Path(args.output); o.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(d/'train.csv'); test=pd.read_csv(d/'test.csv'); sample=pd.read_csv(d/'sample_submission.csv')
    y=train['TARGET'].astype(int).to_numpy(); raw_train=features(train); raw_test=features(test)
    cats=raw_train.select_dtypes(include=['object','string','category']).columns.tolist()
    for c in cats:
        allcats=pd.concat([raw_train[c].astype('string').fillna('__MISSING__'),raw_test[c].astype('string').fillna('__MISSING__')],ignore_index=True).astype('category').cat.categories
        raw_train[c]=pd.Categorical(raw_train[c].astype('string').fillna('__MISSING__'),categories=allcats)
        raw_test[c]=pd.Categorical(raw_test[c].astype('string').fillna('__MISSING__'),categories=allcats)
    X=raw_train.replace([np.inf,-np.inf],np.nan); T=raw_test.replace([np.inf,-np.inf],np.nan)
    # Stratified holdout is for honest model selection and the reported local AUC.
    fold=stratified_folds(y,5); vi=np.flatnonzero(fold==0); ti=np.flatnonzero(fold!=0)
    params={'objective':'binary','metric':'auc','learning_rate':0.025,'num_leaves':15,'max_depth':-1,
      'min_data_in_leaf':80,'feature_fraction':0.85,'bagging_fraction':0.85,'bagging_freq':1,
      'lambda_l1':0.1,'lambda_l2':5.0,'verbosity':-1,'seed':SEED,'feature_fraction_seed':SEED,
      'bagging_seed':SEED,'data_random_seed':SEED,'num_threads':4,'force_col_wise':True,'deterministic':True}
    dtrain=lgb.Dataset(X.iloc[ti],label=y[ti],categorical_feature=cats,free_raw_data=True)
    dvalid=lgb.Dataset(X.iloc[vi],label=y[vi],categorical_feature=cats,reference=dtrain,free_raw_data=True)
    model=lgb.train(params,dtrain,num_boost_round=3000,valid_sets=[dvalid],valid_names=['valid'],
      callbacks=[lgb.early_stopping(160,verbose=False),lgb.log_evaluation(250)])
    pval=model.predict(X.iloc[vi],num_iteration=model.best_iteration)
    valid_auc=auc(y[vi],pval); best=int(model.best_iteration)
    # Refit an equal-fold ensemble on all labelled rows; no test labels or external data are used.
    pred=np.zeros(len(T),dtype=float); counts=np.zeros(len(T),dtype=np.int8); importances=[]
    fold_ids=stratified_folds(y,args.folds,SEED+7)
    for k in range(args.folds):
        v=np.flatnonzero(fold_ids==k); tr=np.flatnonzero(fold_ids!=k)
        ds=lgb.Dataset(X.iloc[tr],label=y[tr],categorical_feature=cats,free_raw_data=True)
        m=lgb.train(params,ds,num_boost_round=max(best,100),callbacks=[lgb.log_evaluation(0)])
        pred+=m.predict(T,num_iteration=max(best,100)); counts+=1
        importances.append(pd.DataFrame({'feature':X.columns,'gain':m.feature_importance(importance_type='gain')}))
        print(f'fold {k+1}/{args.folds} trained; trees={max(best,100)}')
    pred/=counts
    sub=pd.DataFrame({'SK_ID_CURR':test['SK_ID_CURR'],'TARGET':pred})
    # Follow sample column order and ID order exactly.
    sub=sub[sample.columns]; sub.to_csv(o/'Competition_Submission_v2.csv',index=False)
    pd.concat(importances).groupby('feature',as_index=False).gain.mean().sort_values('gain',ascending=False).to_csv(o/'Competition_v2_feature_importance.csv',index=False)
    metrics={'validation_auc':valid_auc,'best_iteration':best,'train_rows':len(train),'test_rows':len(test),'features':X.shape[1],
      'target_rate':float(y.mean()),'validation_rows':len(vi),'seed':SEED,'folds':args.folds,
      'submission_rows':len(sub),'submission_columns':list(sub.columns),'submission_min':float(pred.min()),'submission_max':float(pred.max())}
    (o/'Competition_v2_metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()

