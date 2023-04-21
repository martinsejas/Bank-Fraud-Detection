# Bank Fraud Detection

This notebook trains a sequential neural network to classify fraudulent/legitimate bank transactions. The dataset has around 300k samples, please refer to the .info file for more information on the features and the label. 

Aim of this notebook is to get good results by tuning hyperparemeters and using a meaningful neural network architecture which is well thought out and applies to the use case. 

The results are very good, especially the f-score (recall and precision), however the neural network suffers from a very imbalanced dataset. 

Points of improvement for better results would be: 

- Undersampling for a more balanced dataset. 

- Applying weights on under represented classes

- Threshold grid search.

- More time spent on pre-processing the data, and doing further feature engineering. 

## Running this notebook yourself. 

This notebook uses packages which are standard in Deep Learning applications, and are already pre-installed in [Google Collab](https://colab.research.google.com/). 

If you wish to run it yourself please upload the notebook to google collab, as well as the dataset. 

If not please make sure you have python installed, as well as the right packages. You can install a package by running the following command on your terminal: 


```bash
pip install <package-name>
```

## Neural Network Architecture (Also found in the notebook)

I built my Neural Network using TensorFlow/Keras.
There were many decisions to be made when designing the Neural Network, I will explain why I believe the chosen parameters are the best to tackle the problem.

**Activation Function**: The chosen activation function is SELU, for me this was a straightforward choice as it deals with the vanishing-exploding gradient problem, and performs better than RELU. SELU known to be constraint heavy, but my model fits the requirements, I am dealing with a Sequential Model, have normalized my data, and am making sure to use leCun activation for weights initialization of each layer.

**Output Function**: This is a no-brainer, I have a binary classification problem, my output function is a sigmoid function.

**Controlling Overfitting**: I decided to control overfitting by using Early Stopping. Not only is this an efficient Regularization technique, it will save me time as I explore different architectures for my model. I have decided to monitor validation loss, with a patience of 3. If the validation loss does not improve in 3 epochs I stop the training and restore the best weights.

**Optimizer**: For the optimizer I will use the default AdamW with a varying learning rate.

**Hyperparameter Tuning**: To tune the learning rate I have decided to implement a learning scheduler in my model. I have a starting learning rate of 0.05, that will be halved every time until epoch 6, after this I implement an exponential decay to the learning rate.

**Model Architecture**: For my model architecture I implemented many different configurations. I found that adding anything more than 2 dense layers would make my problem overlycomplex and before badly. Additionally due to the large amount of features of the transformed data set, I found the need of having stacked hidden layers with 1024 neurons each. I found using a binary number to increase the performance of my model.

The accuracy metric may not be appropriate for imbalanced datasets such as this one. I may use precision, recall, and F1 score that take into account the imbalance between the classes.


## Architecture 

![image](https://user-images.githubusercontent.com/99181273/233666362-019cde32-221f-4168-9bb4-06bdb3b8651f.png)


## Loss Graph

![image](https://user-images.githubusercontent.com/99181273/233666503-78a3e3c7-d8f2-4854-899d-7d5060d53b11.png)


## Final Metrics

- Loss: 0.0130
- Precision: 83.6%
- Recall: 85.4% 
- AUC: 0.983

Accuracy is not a reliable in this use case due to the inbalance in the dataset (Fraud cases represent 1.21% of the data), and hence an algorithm that always predicts 'Non-Fraud' would technically be over 98% accurate. 

