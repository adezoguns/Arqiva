import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB



def all_json_to_csv(path_to_json_folder):
    '''This converts all the JSON in the folder into a big CSV'''
    result={}
    dict_network={}

    #Trying to read the element in the json's folder 
    for i, ele in enumerate  (os.listdir(path_to_json_folder)):
        #print(ele)
        with open(path_to_json_folder+"/"+ele, "rb") as f:
            res_json=json.load(f)
            #print(res)
            result[i]=res_json
            print("Processed json_{}".format(i))

    
    #print(result)
    csv_data = pd.DataFrame.from_dict(result, orient='index')
    print("Successfully created the csv")
    csv_data.to_csv("./dirty_data.csv", index=False)

def preprocessing_csv(path_to_csv):
    '''This is the fucntion that converts the dataset into a machine learning format
    removing the attributes we will not use'''
    dict_network={}

    data=pd.read_csv(path_to_csv)
    #print(data.head())

    #Reading the network
    with open("./Challenge/network.adjlist", "r") as f:
        strg=f.read()
    arr=strg.split("\n")  
    #print(arr)  
    for ele in arr:
        #array1.append(ele)
        if len(ele) !=0:
            dict_network[ele.split()[0]]=len(ele.split()) -1
        #print(dict_network)
    arr_node_conn=[]
    noder=data["node_id"]

    for ele in noder:
        arr_node_conn.append(dict_network[ele])

    data["number_node"]=arr_node_conn

    #remove unwanted attributes
    data=data.drop(columns=["fault_id", "node_id", "visit_date:", "original_reported_date", "engineer_note"])
    data["visit_id"] = data["visit_id"].astype(int) + 1
    #print(data.head())

    #print(data.head())
    for i in range(1, 6):
        data=data.replace(to_replace =["LEVEL{}".format(i)], value =i)
    
    #Performing Onehotencoding on node_type, fault_type, engineer_skill_level and outcome
    one_hot = OneHotEncoder(sparse=False, handle_unknown='ignore')
    train_enc = one_hot.fit_transform(data[["node_type","fault_type", "outcome"]])
    data_enco=pd.DataFrame(train_enc, columns=one_hot.get_feature_names())

    #Performing MinMaxScaling on column visit_id and node_age
    scaler = MinMaxScaler()
    data_scaled  = scaler.fit_transform(data[["visit_id", "node_age", "engineer_skill_level", "number_node"]]) 
    
    #This block combines the header from the scaled and onehotencoding together
    onehot_header=list(one_hot.get_feature_names().copy())
    data_header=["visit_id", "node_age", "engineer_skill_level", "number_node"]
    data_header.extend(onehot_header)

    #Combining the scaled and the onehotencoded columns together
    combo_data = np.concatenate([data_scaled, data_enco], axis=1)
    
    combo_data = pd.DataFrame(combo_data)

    #Changing the header back to a readable format from the 0,1,2 ... format generated after encoding
    combo_data = combo_data.rename(columns={ i: data_header[i] for i in range (len(data_header))})

    #Dropping the extra column for the binary outcome FAIL
    combo_data = combo_data.drop(columns=["x2_FAIL"])
    
    # arr=combo_data["x3_SUCCESS"]
    # count=0
    # for ele in arr:
    #     if ele==1:
    #         count+=1
    # print(count)


    print("\nSucessfully preprocessed the dataset and created a machine learning ready csv")

    combo_data.to_csv("./ml_ready.csv", index=False)

   

def modelling(path_to_csv2):
    '''Modelling the dataset with four different ML algorithms'''

    result_dict={}
    #Colour for the models
    #colour=["b", "y", "g", "c", "m", "k"]
    colour=['b', 'navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal']
    colour_counter=0

    #Reading the the preprocessed csv data
    data1=pd.read_csv(path_to_csv2)
    #Converting it into a dataframe
    data1=pd.DataFrame(data1)

    #Extracting X and y from the dataset
    X=data1.values[:, 0: -1]
    y=data1.values[:, -1]

    # Divide the the dataset into test train split
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42, shuffle=True)

    #Trying out four different ML algorithms
    model={"LR":LogisticRegression(solver='lbfgs'), "RF":RandomForestClassifier() ,"DTC":DecisionTreeClassifier(), "XGB":XGBClassifier(), "SVM":SVC(), "NB":GaussianNB()}

    #Iterating to fit the model
    for ele in model.keys():
        model[ele].fit(X_train, y_train)
        pred=model[ele].predict(X_test)
        accu= accuracy_score(y_test, pred)
        result_dict[ele]=pred
        print("Accuracy of the model {} is {:.3f}%".format(ele, accu*100))
    
    print("\n")
    #Iterating to check each classification report
    for ele in result_dict:
        
        class_report= classification_report(y_test, result_dict[ele])
        print("\nClassification report {}".format(ele))
        print("{}".format(class_report))


    #Iterating to plot ROC curve
    for ele in result_dict:
    
        fpr, tpr, threshold = roc_curve(y_test, result_dict[ele])
        roc_auc = auc(fpr, tpr)

        #Plotting
        plt.title('ROC curve')
        plt.plot(fpr, tpr, colour[colour_counter], label = 'AUC for {} = %0.3f'.format(ele) % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1.0])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        colour_counter+=1
    plt.tight_layout()
    plt.savefig('./ROC.pdf', format='pdf', dpi=150)
    plt.cla()
    plt.clf()


def rfc_feature_ranking(path_to_csv2):
    """Using Random Forest Classifier to rank features"""

    #Using pandas to read the features
    data1=pd.read_csv(path_to_csv2)

    #Getting the header of each column for  later use
    data_header= data1.columns
    #Converting it into dataframe
    data1=pd.DataFrame(data1)
    X=data1.values[:, 0: -1]
    y=data1.values[:, -1]
    #Calling RFC
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X, y)
    rank = rf.feature_importances_
   
    forest_importances = pd.Series(rank, index=data_header[:-1])
    #Plotting
    fig, ax = plt.subplots()
    forest_importances.plot.bar(ax=ax)
    ax.set_title("Feature ranking using RFC")
    ax.set_ylabel("Ranking")
    plt.grid()
    #fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    #plt.show()
    plt.savefig('./RFC.pdf', format='pdf', dpi=150, bbox_inches='tight')
    plt.cla()
    plt.clf()


def rfe_feature_ranking(path_to_csv2):
    """This function uses RFE to rank features"""

    #Using pandas to read the ml ready csv
    data1=pd.read_csv(path_to_csv2)
    #The header
    data_header= data1.columns
    data1=pd.DataFrame(data1)
    X=data1.values[:, 0: -1]
    y=data1.values[:, -1]

    # Divide the the dataset into test train split
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42, shuffle=True)


    #Calling logistic regression
    model = LogisticRegression()
    #model= XGBClassifier()
    #model = ExtraTreesClassifier( random_state=seed)
    rfe = RFE(model, 1)
    fit = rfe.fit(X_train, y_train)
    #print("Num Features: {}".format(fit.n_features_))
    #print("Selected Features: {}".format(fit.support_))
    #print("Feature Ranking: {}".format(fit.ranking_))

    #Ranking
    rank=fit.ranking_

    x= [i for i in range(len(data_header)-1)]

    #Plotting
    plt.xticks(x, data_header[:-1])
    plt.xticks(range(len(data_header)-1), data_header[:-1], rotation=90)
    plt.bar(x, rank)
    plt.grid()
    #plt.ylim((0.0,0.4))
    plt.xlabel('Features')
    plt.ylabel('Ranking')
    plt.title("The RFE ranking for features according to importance")
    #plt.tight_layout()
    plt.savefig('./RFE.pdf', format='pdf', dpi=150, bbox_inches='tight')
    plt.cla()
    plt.clf()


def etc_feature_ranking(path_to_csv2):
    """This function uses Extra Tree Classifier to rank feartures"""
    # load data 
    seed=110 
    #Using pandas to read csv
    data1=pd.read_csv(path_to_csv2)

    #Readig the column names
    data_header= data1.columns
    data1=pd.DataFrame(data1) 

    #Split dataset in X and y
    X=data1.values[:, 0: -1]
    y=data1.values[:, -1]

    # Divide the the dataset into test train split
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42, shuffle=True)

    # feature extraction
    #Calling the ETC
    model = ExtraTreesClassifier(n_estimators=5, random_state=seed)
    #Fitting the model
    model.fit(X_train, y_train)
    #Ranking
    rank =model.feature_importances_
    x= [i for i in range(len(data_header)-1)]

    #Ploting the model
    plt.xticks(x, data_header[:-1])
    plt.xticks(range(len(data_header)-1), data_header[:-1], rotation=90)
    plt.bar(x, rank)
    plt.grid()
    #plt.ylim((0.0,0.4))
    plt.xlabel('Features')
    plt.ylabel('Ranking')
    plt.title("The extra tree classifier ranking for features according to importance")
    #plt.tight_layout(pad=0.9, w_pad=0.5, h_pad=1.0)
    plt.savefig('./ETC.pdf', format='pdf', dpi=150, bbox_inches='tight')
    plt.cla()
    plt.clf()


def us_feature_ranking(path_to_csv2):
    # load data 
    seed=110 

    #Read csv
    data1=pd.read_csv(path_to_csv2)
    #Get the column names
    data_header= data1.columns
    #Convert it to dataframe
    data1=pd.DataFrame(data1) 

    #Extract Xand y from the dataset
    X=data1.values[:, 0: -1]
    y=data1.values[:, -1]

    # Divide the the dataset into test train split
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42, shuffle=True)


    # feature extraction
    test = SelectKBest(score_func=chi2, k=10)
    fit = test.fit(X_train, y_train)
    # summarize scores
    np.set_printoptions(precision=4)
    rank= fit.scores_
    #rank = fit.transform(X)

    x= [i for i in range(len(data_header)-1)]

    #Plotting
    plt.xticks(x, data_header[:-1])
    plt.xticks(range(len(data_header)-1), data_header[:-1], rotation=90)
    plt.bar(x, rank)
    plt.grid()
    #plt.ylim((0.0,0.4))
    plt.xlabel('Features')
    plt.ylabel('Ranking')
    plt.title("The US ranking for features according to importance")
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('./US.pdf', format='pdf', dpi=150, bbox_inches='tight')
    plt.cla()
    plt.clf()
    

if  __name__=="__main__":

    #all_json_to_csv("/home/adeola/Arqiva/Challenge/visits")
    preprocessing_csv("./dirty_data.csv")
    modelling("./ml_ready.csv")
    rfc_feature_ranking("./ml_ready.csv")
    rfe_feature_ranking("./ml_ready.csv")
    etc_feature_ranking("./ml_ready.csv")
    us_feature_ranking("./ml_ready.csv")