import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report,f1_score
from sklearn.impute import KNNImputer
import pickle

def ml():
    df = pd.read_csv("archive.zip", compression = "zip")

    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    x_train,x_test,y_train,y_test = train_test_split(x,y, stratify = y, test_size = 0.20, random_state = 36)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    imputer = KNNImputer(n_neighbors = 5)
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)


#model 1
    model1 = LogisticRegression()                                         
    model1.fit(x_train,y_train)

    model1_train_ypred = model1.predict(x_train)
    model1_train_f1score = f1_score(y_train,model1_train_ypred)
    
    model1_test_ypred = model1.predict(x_test)
    model1_test_f1score = f1_score(y_test,model1_test_ypred)

    # model1_train_f1score = 0.7410714285714286, model2_test_f1score = 0.7675675675675676

#model 2
    model2 = LinearSVC(max_iter = 5000)
    model2.fit(x_train,y_train)

    model2_train_ypred = model2.predict(x_train)
    model2_train_f1score = f1_score(y_train,model2_train_ypred)

    model2_test_ypred = model2.predict(x_test)
    model2_test_f1score = f1_score(y_test,model2_test_ypred)

    # model2_train_f1score = 0.7166921898928025, model2_test_f1score = 0.7344632768361582

#model 3
    model3 = DecisionTreeClassifier(criterion = "gini", max_depth = 8, random_state = 36)
    model3.fit(x_train,y_train)

    model3_train_ypred = model3.predict(x_train)
    model3_train_f1score = f1_score(y_train,model3_train_ypred)

    model3_test_ypred = model3.predict(x_test)
    model3_test_f1score = f1_score(y_test,model3_test_ypred)

    # model3_train_f1score = 0.9002770083102493, model3_test_f1score = 0.8421052631578947

#model 4
    model4 = RandomForestClassifier(n_estimators = 200, min_samples_split = 10, max_depth = 9, random_state = 36)
    model4.fit(x_train,y_train)

    model4_train_ypred = model4.predict(x_train)
    model4_train_f1score = f1_score(y_train,model4_train_ypred)

    model4_test_ypred = model4.predict(x_test)
    model4_test_f1score = f1_score(y_test,model4_test_ypred)

    # model4_train_f1score = 0.8857938718662952, model4_test_f1score = 0.8631578947368421

#model 5
    model5 = XGBClassifier(n_estimators = 500, learning_rate = 0.1 , max_depth = 4, random_state = 36)
    model5.fit(x_train,y_train)

    model5_train_ypred = model5.predict(x_train)
    model5_train_f1score = f1_score(y_train,model5_train_ypred)

    model5_test_ypred = model5.predict(x_test)
    model5_test_f1score = f1_score(y_test,model5_test_ypred)

    # model5_train_f1score = 1.0, model5_test_f1score = 0.8817204301075269

#model 6
    model6 = GaussianNB()
    model6.fit(x_train,y_train)

    model6_train_ypred = model6.predict(x_train)
    model6_train_f1score = f1_score(y_train,model6_train_ypred)

    model6_test_ypred = model6.predict(x_test)
    model6_test_f1score = f1_score(y_test,model6_test_ypred)

    # model6_train_f1score = 0.11355246967832659, model6_test_f1score = 0.11588275391956374

    models = {
        "Logistic Regression": model1_test_f1score,
        "Linear SVC": model2_test_f1score,
        "Decision Tree Classifier": model3_test_f1score,
        "Random Forest Classifier": model4_test_f1score,
        "XGB Classifier": model5_test_f1score,
        "GaussianNB": model6_test_f1score
    }

    best_model_name = None
    best_score = 0

    for model_name, f1 in models.items():
        if f1 > best_score:
            best_model_name = model_name
            best_score = f1


    model_dict = {
        "Logistic Regression": model1,
        "Linear SVC": model2,
        "Decision Tree Classifier": model3,
        "Random Forest Classifier": model4,
        "XGB Classifier": model5,
        "GaussianNB": model6
    }
    best_model = model_dict[best_model_name]


    with open("best_model.pkl", "wb") as file:
        pickle.dump(best_model, file)
    with open("scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)
    with open("imputer.pkl", "wb") as file:
        pickle.dump(imputer, file)

    
    return {
    "best_model_name": best_model_name,
    "best_score": best_score,
    "Logistic Regression": model1_test_f1score,
    "Linear SVC": model2_test_f1score,
    "Decision Tree": model3_test_f1score,
    "Random Forest": model4_test_f1score,
    "XGB Classifier": model5_test_f1score,
    "GaussianNB": model6_test_f1score
    }

ml()


    





  




    




    

   





