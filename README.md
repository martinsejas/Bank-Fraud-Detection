# Bank Fraud Detection

This notebook trains a sequential neural network to classify fraudulent/legitimate bank transactions. The dataset has around 300k samples, please refer to the .info file for more information on the features and the label. 

Aim of this notebook is to get good results by tuning hyperparemeters and using a meaningful neural network architecture which is well thought out and applies to the use case. 

The results are very good, especially the f-score (recall and precision), however the neural network suffers from a very imbalanced dataset. 

Points of improvement for better results would be: 

- Undersampling for a more balanced dataset. 

- More time spent on pre-processing the data, and doing further feature engineering. 

## Running this notebook yourself. 

This notebook uses packages which are standard in Deep Learning applications, and are already pre-installed in [Google Collab](https://colab.research.google.com/). 

If you wish to run it yourself please upload the notebook to google collab, as well as the dataset. 

If not please make sure you have python installed, as well as the right packages. You can install a package by running the following command on your terminal: 


```bash
pip install <package-name>
```
