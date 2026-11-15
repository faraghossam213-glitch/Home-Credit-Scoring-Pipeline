HOME CREDIT COMPETITION — REPRODUCIBLE SUBMISSION

Competition_Submission.csv was generated only from the supplied train.csv and test.csv. It has 61,500 rows and columns SK_ID_CURR,TARGET. The local stratified holdout ROC-AUC was 0.7570. Final predictions are the mean of five LightGBM models.

To reproduce: extract the course Competition archive, install Python 3.10+ plus pandas, numpy and LightGBM 4.7.0, then run:
  python competition_model.py --data-dir "Competition/input" --output output --folds 5
Upload output/submission.csv to Omnicampus. Check the uploaded scoring timestamp. The course README says final rank uses the last submission and final scoring uses the private leaderboard; keep this file as your last submission only if you want this version considered.

The four Competition Week .crdownload files were incomplete at inspection; this submission uses the complete Competition archive provided separately.
