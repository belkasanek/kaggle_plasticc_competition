# [Kaggle PLAsTiCC Astronomical Classification Competition](https://www.kaggle.com/c/PLAsTiCC-2018)
 The tasks was to classify astronomical sources that vary with time into 15 different classes, scaling from a small training set (~8000 objects) to a very large test set (~1.5 million). Submissions are evaluated using a weighted multi-class logarithmic loss. The overall effect is such that each class is   roughly equally important for the final score. More background information is available in this [paper](https://arxiv.org/abs/1810.00001).
 
 5 LightGBM model and 3 XGBoost model were trained on 76 features and stacked to obtain score bellow.
|public score|private score|final rank| 
|---|---|---|
|0.8913|0.9087| 32th (*top3%*)|

