"""Company A churn analysis and reproducible proposal evidence.
Run: python telecom_analysis.py --data-dir ./telecom --output ./output
"""
from pathlib import Path
import argparse,json,numpy as np,pandas as pd
import lightgbm as lgb
SEED=42

def auc(y,p):
    y=np.asarray(y,dtype=np.int8); r=pd.Series(np.asarray(p)).rank(method='average').to_numpy(); n1=y.sum();n0=len(y)-n1
    return float((r[y==1].sum()-n1*(n1+1)/2)/(n1*n0))
def average_precision(y,p):
    y=np.asarray(y,dtype=np.int8); order=np.argsort(-np.asarray(p)); ys=y[order]; prec=np.cumsum(ys)/(np.arange(len(ys))+1)
    return float((prec*ys).sum()/max(ys.sum(),1))
def stratified_split(y,valid_fraction=.25,seed=SEED):
    rng=np.random.default_rng(seed); a=[];b=[]
    for c in np.unique(y):
        idx=np.flatnonzero(y==c);rng.shuffle(idx); n=int(round(len(idx)*valid_fraction));b.extend(idx[:n]);a.extend(idx[n:])
    return np.array(a),np.array(b)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',default='telecom');ap.add_argument('--output',default='output');a=ap.parse_args()
    d=Path(a.data_dir);o=Path(a.output);o.mkdir(parents=True,exist_ok=True)
    client=pd.read_csv(d/'Client.csv');record=pd.read_csv(d/'Record.csv')
    if client.Customer_ID.duplicated().any() or record.Customer_ID.duplicated().any(): raise ValueError('Expected one row per Customer_ID in each input table')
    df=record.merge(client,on='Customer_ID',how='inner',validate='one_to_one')
    y=df['churn'].astype(int).to_numpy();X=df.drop(columns=['churn','Customer_ID']).copy()
    # Compact, interpretable ratios derived from fields available at the observation date.
    X['MEAN_MONTHLY_REVENUE_PER_TENURE']=X['totrev']/(X['months'].clip(lower=1))
    X['EQUIPMENT_AGE_PER_TENURE']=X['eqpdays']/(X['months'].clip(lower=1))
    X['RECENT_TO_LIFETIME_REVENUE']=X['avg3rev']/(X['avgrev'].replace(0,np.nan))
    X['RECENT_TO_LIFETIME_USAGE']=X['avg3mou']/(X['avgmou'].replace(0,np.nan))
    cats=X.select_dtypes(include=['object','string','category']).columns.tolist()
    for c in cats:X[c]=X[c].astype('string').fillna('__MISSING__').astype('category')
    X=X.replace([np.inf,-np.inf],np.nan)
    tr,va=stratified_split(y,.25)
    params={'objective':'binary','metric':'auc','learning_rate':.025,'num_leaves':31,'max_depth':-1,'min_data_in_leaf':120,
      'feature_fraction':.85,'bagging_fraction':.85,'bagging_freq':1,'lambda_l1':.1,'lambda_l2':3.,'verbosity':-1,'seed':SEED,
      'feature_fraction_seed':SEED,'bagging_seed':SEED,'data_random_seed':SEED,'num_threads':4,'force_col_wise':True,'deterministic':True}
    ds=lgb.Dataset(X.iloc[tr],label=y[tr],categorical_feature=cats,free_raw_data=True);dv=lgb.Dataset(X.iloc[va],label=y[va],categorical_feature=cats,reference=ds,free_raw_data=True)
    model=lgb.train(params,ds,num_boost_round=3000,valid_sets=[dv],valid_names=['valid'],callbacks=[lgb.early_stopping(160,verbose=False),lgb.log_evaluation(250)])
    prob=model.predict(X.iloc[va],num_iteration=model.best_iteration);yv=y[va]
    rank=np.argsort(-prob);topn=max(1,int(np.ceil(.20*len(prob))));top=rank[:topn];overall=float(yv.mean());top_rate=float(yv[top].mean());capture=float(yv[top].sum()/max(yv.sum(),1))
    rev=pd.to_numeric(df.iloc[va]['rev_Mean'],errors='coerce').to_numpy(); avg_rev=float(np.nanmean(rev[top]))
    # Illustrative only: dataset currency, 12-month horizon, $10-equivalent cost, no margin included.
    unit_cost=10.;horizon=12
    scenarios=[]
    for uplift in [.01,.03,.05]:
        incremental_revenue=avg_rev*horizon*uplift
        scenarios.append({'assumed_incremental_retention_pp':uplift*100,'gross_revenue_saved_per_target':incremental_revenue,
          'assumed_offer_cost_per_target_dataset_units':unit_cost,'net_revenue_proxy_per_target':incremental_revenue-unit_cost,
          'break_even_retention_uplift_pp':100*unit_cost/(avg_rev*horizon) if avg_rev>0 else None})
    metrics={'rows':int(len(df)),'features':int(X.shape[1]),'categorical_features':len(cats),'churn_rate':overall,'validation_rows':int(len(va),),
      'validation_auc':auc(yv,prob),'average_precision':average_precision(yv,prob),'best_iteration':int(model.best_iteration),
      'top20_count':int(topn),'top20_churn_rate':top_rate,'top20_capture_rate':capture,'top20_lift_vs_base':top_rate/overall,
      'top20_mean_monthly_revenue_dataset_units':avg_rev,'business_scenarios':scenarios,
      'seed':SEED,'model':'LightGBM native categorical binary classifier'}
    (o/'telecom_metrics.json').write_text(json.dumps(metrics,indent=2))
    imp=pd.DataFrame({'feature':X.columns,'gain':model.feature_importance(importance_type='gain'),'split':model.feature_importance(importance_type='split')}).sort_values('gain',ascending=False)
    imp.to_csv(o/'telecom_feature_importance.csv',index=False)
    pd.DataFrame({'actual_churn':yv,'predicted_churn_probability':prob,'monthly_revenue_dataset_units':rev}).to_csv(o/'telecom_validation_predictions.csv',index=False)
    miss=df.isna().mean().sort_values(ascending=False).rename('missing_fraction').reset_index().rename(columns={'index':'feature'});miss.to_csv(o/'telecom_missingness.csv',index=False)
    print(json.dumps(metrics,indent=2));print('\nTop features by gain:\n',imp.head(15).to_string(index=False))
if __name__=='__main__':main()
