# Credit_Risk_Analysis


## Overview of Analysis
Using Python to build and evaluate various machine learning models to predict credit risk. Oversampled the data using RandomOverSampler and SMOTE algorithms, undersampled the data using the ClusterCentroids algorithm, and compared two machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier. After evaluating these models, I will make a recommendation on whether they should be used to predict credit risk. 


## Resources
- Data: 
  - [LoanStats_2019Q1.csv]("../resources/LoanStats_2019Q1.csv")
- Software: 
  - Python
  - Anaconda
  - Jupyter Notebook


## Results
### SMOTE Oversampling
![resources/score_smote_oversampling.png](resources/score_smote_oversampling.png)
The balanced accuracy score is 61.3%.

![resources/report_oversampling.png](resources/report_oversampling.png)
Because of the high number of the low_risk population, the precision is almost 100% with a sensitivity of 66%.

![resources/matrix_smote_oversampling.png](resources/matrix_smote_oversampling.png)


### ClusterCentroids Model
![resources/score_cluster_centroid.png](resources/score_cluster_centroid.png)
The balanced accuracy score is 51%.

![resources/report_cluster_centroid.png](resources/report_cluster_centroid.png)
Because of the high number of false positives, the low_risk sensitivity is about 44% for this model, with a precision of about 100%. 

![resources/matrix_cluster_centroid.png](resources/matrix_cluster_centroid.png)

### Ease Ensemble AdaBoost Classifier
![resources/score_easy_ensemble_adaboost_classifier.png](resources/score_easy_ensemble_adaboost_classifier.png)
This balanced accuracy score is high, about 93.2%.

![resources/report_ease_ensemble_adaboost_classifier.png](resources/report_ease_ensemble_adaboost_classifier.png)
Because of a low number of false positives, the low_risk sensitivity is 92% with 100% precision. 

![resources/confusion_easy_ensemble_adaboost_classifier.png](resources/confusion_easy_ensemble_adaboost_classifier.png)

### BalancedRandomForestClassifier Model
![resources/confusion_matrix_balanced_random_forest_classifier.png](resources/confusion_matrix_balanced_random_forest_classifier.png)
The balanced accuracy score is about 79%.

![resources/confusion_matrix_balanced_random_forest_classifier.png](resources/confusion_matrix_balanced_random_forest_classifier.png)
The high risk precision is about 3% with a sensitivity of 70%. Due to the high number of the low_risk population, the precision is almost 100% with a sensitivity of 87%.

![resources/confusion_matrix_balanced_random_forest_classifier.png](resources/confusion_matrix_balanced_random_forest_classifier.png)

### Naive Random Oversampling
![resources/score_naive_random_oversampling.png](resources/score_naive_random_oversampling.png)
The balanced accuracy score is about 61.4%.

![resources/report_naive_random_oversampling.png](resources/report_naive_random_oversampling.png)
The high risk precision is about 1% with a sensitivity of 61%, while the low risk precision is 100% with a sensitivity of about 62%.

![resources/matrix_naive_random_oversampling.png](resources/matrix_naive_random_oversampling.png)

### Combo - Over and Under Sampling
![resources/score_combo_over_under_sampling.png](resources/score_combo_over_under_sampling.png)
The balanced accuracy score is 64%.

![resources/report_combo_over_under_sampling.png](resources/report_combo_over_under_sampling.png)
The high risk precision is 1% with a sensitivity of 70%, while the low risk precision is 100% with a sensitivity of 58%.

![resources/matrix_combo_over_under_sampling.png](resources/matrix_combo_over_under_sampling.png)


## Summary
Every model I evaluated to predict credit risk has shown weak precision. The models that had more improvement on the sensitivity of high risk credits was the Ensemble models. The EasyEnsembleClassifier model has a recall of 92%, detecting almost all high risk credit. However, with low precision, a lot of low risk credits are falsely detected as high risk, which could hurt the bank by missing those opportunities. Because of this specific situation and the models showing weak precision, I would not recommend any of these models to predict credit risk to the bank. For a bank, you want to have high precision so that you can strategize opportunities to increase revenue. 
