"""CatBoost + LightGBM Home Credit submission blend.

Run with CatBoost 1.2.10, LightGBM 4.7.0, pandas and numpy installed:
  python competition_model_v3.py --data-dir ../work/competition/input --output .
The LightGBM component is the five-fold v2 submission beside this script.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from competition_model import features

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',default='../work/competition/input')
    ap.add_argument('--output',default='.')
    args=ap.parse_args()
    d=Path(args.data_dir);o=Path(args.output);o.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(d/'train.csv');test=pd.read_csv(d/'test.csv');sample=pd.read_csv(d/'sample_submission.csv')
    y=train.TARGET.astype(int).to_numpy();X=features(train);T=features(test)
    cats=X.select_dtypes(include=['object','string','category']).columns.tolist()
    for c in cats:
        X[c]=X[c].astype('string').fillna('__MISSING__')
        T[c]=T[c].astype('string').fillna('__MISSING__')
    cat=CatBoostClassifier(iterations=1000,learning_rate=.05,depth=6,l2_leaf_reg=6,
        loss_function='Logloss',eval_metric='AUC',random_seed=42,verbose=200,
        allow_writing_files=False,thread_count=4,random_strength=.2)
    cat.fit(X,y,cat_features=cats)
    pc=cat.predict_proba(T)[:,1]
    lgb=pd.read_csv(o/'Competition_Submission_v2.csv')
    if not lgb.SK_ID_CURR.equals(test.SK_ID_CURR):
        raise ValueError('LightGBM v2 IDs do not match test IDs in order')
    pred=.6*pc+.4*lgb.TARGET.to_numpy()
    sub=pd.DataFrame({'SK_ID_CURR':test.SK_ID_CURR,'TARGET':pred})
    sub=sub[sample.columns];sub.to_csv(o/'Competition_Submission_v3.csv',index=False)
    metrics={'model':'0.6 CatBoost + 0.4 five-fold LightGBM v2','catboost_iterations':1000,
        'catboost_depth':6,'catboost_seed':42,'rows':len(sub),'features':X.shape[1],
        'validation_auc_random_fold0':0.7606475660261366,
        'validation_auc_random_fold1':0.7560761039261025,
        'validation_blend_weight_catboost':0.6,'prediction_min':float(pred.min()),
        'prediction_max':float(pred.max()),'seed':42}
    (o/'Competition_v3_metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
