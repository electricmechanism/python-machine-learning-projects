# python-machine-learning-projects

1) NLP-model-comments-classification.ipynb
 
Objective of the project:
Online store "Wikishop" launches a new service. Now users can edit and supplement product descriptions, just like in wiki communities. That is, clients propose their edits and comment on the changes of others. The store needs a tool that will look for toxic comments and submit them for moderation. Train the model to classify comments into positive and negative. At your disposal is a dataset with markup on the toxicity of edits. Build a model with a quality metric F1 value of at least 0.75.

Conclusion:
Loaded text. Carried out pre-processing: Carried out cleaning and lemming of the text. We checked the ratio of classes in the target trait for balance. Found that the classes are unbalanced. The classes were balanced using the upsample and downsample technology. We transform the training and test samples using the TF-IDF Vectorizer algorithm - we extract features from the text. 
Trained and explored three models:  
Naive Bayes Model,  
LogisticRegression,  
DecisionTreeClassifier.  
The best result on the test sample was shown by the LogisticRegression model with f1 = 0.78.  
The goal of the project has been achieved. A model was built with the value of the quality metric F1 not less than 0.75.

The project is written in the language Python.

The project file can be opened with Python Jupyter Notebook.

Project status:
Finished.

2) car-cost-forecasting.ipynb

Objective of the project:
Service for the sale of used cars "Not beaten, not beautiful" is developing an application to attract new customers. In it, you can quickly find out the market value of your car. At your disposal are historical data: technical specifications, equipment and prices of cars. You need to build a model to determine the cost.

Conclusion:
The results of prediction of the best LGBMRegressor model on the test set: -RMSE on the test set - 1162.30, -Prediction rate - 4.09s. Project goal achieved, RMSE received less than 2500.

The customer is important:

quality of prediction;
prediction speed;
studying time.
The project is written in the language Python.

The project file can be opened with Python Jupyter Notebook.

Project status:
Finished.

3) forecasting-bank-clients.ipynb

Objective of the project:
Customers began to leave Beta-Bank. Every month. A little, but noticeable. Banking marketers figured it was cheaper to keep current customers than to attract new ones.
It is necessary to predict whether the client will leave the bank in the near future or not. You are provided with historical data on customer behavior and termination of agreements with the bank.
Build a model with an extremely large F1-measure. To pass the project successfully, you need to bring the metric to 0.59. Check the F1-measure on the test set yourself.
Additionally measure AUC-ROC, compare its value with F1-measure.
Data Source: https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling

Conclusion:
Prepared data for model building and research. Passes processed. Column names have been changed.
The data has been checked for duplicates. No duplicates found. Removed unnecessary column features: 'rownumber', 'customerid',
'surname'. Checked the data for outliers. No outliers found.
The data was checked for multicollinearity. The phenomenon of multicollinearity in the data is not exposed.
We encoded the data and got rid of categorical features using the One hot encoder (OHE) method.  
We divided the initial data into three samples: training, validation and test. We allocated 60% of the data for the training set, and 20% for the validation and test sets. We received three data sets df_train for training the model, df_valid for checking the model for retraining, df_test for assessing the quality and accuracy of the model.
We checked how often the class "1" or "0" occurs in the target feature of our original dataset.
The ratio is unbalanced: 0 ~ 80% and 1 ~ 20%. The difference is 4 times. 
For the Random Forest Classifier model, the best F1-measure is 0.6370023419203746 for a tree depth of 13 and an estimators of 135. The F1-measure is improved over
with the result of training the model without taking into account the imbalance. The parameters of the best tree depth model and estimators have changed compared to the unbalanced model. Thus, taking into account the imbalance in the construction of the model affects the choice of the optimal parameters of the model. <br>   
The LogisticRegression, DecisionTreeClassifier, RandomForestClassifier models are analyzed.  
Best Model: Random Forest Classifier.
For the Random Forest Classifier model, the best F1-measure is 0.6370023419203746 for a tree depth of 13 and
parameter estimators equal to 135.
The F1-measure value has improved compared to
with the result of training the model without taking into account the imbalance. The upsample method gives a better show of F1-measures compared to
with the downsample method: the value of the F1-measure increased by 0.3 when testing on the validation set.  
For the best Random Forest Classifier model, it was tested on a test sample. Result: F1-measure = 0.6096385542168674. It was possible to achieve an F1-measure of at least 0.59 according to the project assignment.

The project is written in the language Python.

The project file can be opened with Python Jupyter Notebook.

Project status:
Finished.

4) forecasting-taxi-orders.ipynb
 
Objective of the project:
Forecasting Taxi Orders
The Clear Taxi company has collected historical data on taxi orders at airports. To attract more drivers during the peak period, you need to predict the number of taxi orders for the next hour. Build a model for such a prediction.
The value of the RMSE metric on the test sample should be no more than 48.
We need:
Load the data and resample it one hour at a time. Analyze data. Train different models with different hyperparameters. Make a test sample of 10% of the original data. Check the data on the test sample and draw conclusions. The data is in the taxi.csv file. The number of orders is in the num_orders column (from the English number of orders, “number of orders”).

Conclusion:
The results of predicting the best LGBMRegressor model on the test set: -RMSE on the test set- 40.89. The goal of the project was achieved, the RMSE obtained on the test sample was less than 48.

The project is written in the language Python.

The project file can be opened with Python Jupyter Notebook.

Project status:
Finished.

5) project_vc.ipynb

Objective of the project:
Determining a person's age from his photo. A chain supermarket introduces a computer vision system to process customer photos. Photo fixation in the checkout area will help determine the age of customers in order to:
Analyze purchases and offer products that may be of interest to buyers of this age group;
Control the conscientiousness of cashiers when selling alcohol.
Build a model that will determine the approximate age of a person from a photograph.
At your disposal is a set of photographs of people with age indication.

Conclusion:
To solve the problem and build the model, a network with the RestNet50 architecture was used, with the number of epochs 10.
For some categories of ages, there are significantly fewer photographs.
The distribution schedule is close to normal.
The data set contains the most photographs of people between the ages of 10 and 50. There are much fewer photographs of other ages, which can adversely affect the quality of education.
The photo dataset contains photos on a black background and rotated at different angles to the right and left about the x-axis, as well as photos taken under different lighting conditions, as well as black and white photos. Such a diverse set of photographs can have a positive effect on the quality of model training, as the model will learn to recognize age in photographs under different street, with any lighting, as well as in black and white photographs.
The goal of the project has been achieved, the MAE of the model is no more than 8.

The project is written in the language Python.

The project file can be opened with Python Jupyter Notebook.

Project status:
Finished.

6) CV1_MNIST.ipynb

Objective of the project:
MNIST fashion classification. the MNIST set consists of 28x28 images, each pixel of which represents a shade of gray. The dataset contains images of t-shirts, tops, sandals, and even boots. Our task is to create a neural network that receives these 784 bytes at the input, and at the output returns to which category of clothes out of 10 available the element submitted at the input belongs to.

Conclusion:
As the model is trained, the loss function value and accuracy metric are displayed for each training iteration. This model achieves an accuracy of around 0.87 (87%) on the training data.

The project is written in the language Python.

The project file can be opened with Python Jupyter Notebook.

Project status:
Finished.

7) NSL_KDD_v2.ipynb
   
We analyze traffic from the SDN network and detect a network attack using the NSL-KDD dataset. NSL-KDD is a public dataset, which has been developed from the earlier KDD cup99 dataset (Tavallaee et al., 2009). In ours, several machine learning models, including deep learning neural networks, are trained on the NSL-KDD dataset, and we select the best model based on various metrics.
The data used to train and test the network is also included in the project:
input

9) Model_prediction_2.ipynb
    
In this project, we study the fault tolerance of SDN networks by examining the current network traffic for the presence of traffic anomalies that can disrupt the operation of the SDN network.
The data used to train and test the network is also included in the project:
M_Dataset_2_30.csv
M_Dataset_2_70.csv

