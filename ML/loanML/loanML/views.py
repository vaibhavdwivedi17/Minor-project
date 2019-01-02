from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import redirect
from django.template.response import TemplateResponse
# Create your views here.'''
import numpy as np
import pandas as pd

def predictor():

    import pandas as pd
    import numpy as np   #For mathematical calculations
    import seaborn as sns   #For data visualization
    import matplotlib.pyplot as plt    #For plotting graphs
    #matplotlib inline
    import warnings      #To ignore any warnings
    #rnings.filterwarnings("ignore")
    #Reading data from respective data files
    train = pd.read_csv("Train.csv")
    # test = pd.read_csv("Test.csv")
    #Making copies, not to lose the originals
    train_original=train.copy()
    # test_original=test.copy()
    train['Dependents'].replace('3+', 3,inplace=True)
    # test['Dependents'].replace('3+', 3,inplace=True)
    train['Loan_Status'].replace('N', 0,inplace=True)
    train['Loan_Status'].replace('Y', 1,inplace=True)
    train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
    train['Married'].fillna(train['Married'].mode()[0], inplace=True)
    train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
    train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
    train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
    train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
    train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
    # test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
    # test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
    # test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
    # test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
    # test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
    # test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
    train['LoanAmount_log'] = np.log(train['LoanAmount'])
    # train['LoanAmount_log'].hist(bins=20)
    # test['LoanAmount_log'] = np.log(test['LoanAmount'])
    train=train.drop('Loan_ID',axis=1)
    # train=train.drop('Property_Area',axis=1)
    # test=test.drop('Loan_ID',axis=1)
    X = train.drop('Loan_Status',1)
    y = train.Loan_Status
    X=pd.get_dummies(X)
    train=pd.get_dummies(train)
    # test=pd.get_dummies(test)
    from sklearn.model_selection import train_test_split
    x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)
    from sklearn.linear_model import LogisticRegression
    # from sklearn.metrics import accuracy_score
    model = LogisticRegression()
    model.fit(x_train, y_train)
    print('predector  executed')
    return model
def train_model(request):
    global g_model
    g_model=predictor()
    return redirect('/')

def ml(request):

    suff_list={
        'Property_Area':['Property_Area_Rural' ,'Property_Area_Urban','Property_Area_Semiurban'],
        'Dependents':['Dependents_0','Dependents_1','Dependents_2','Dependents_3'] ,
        'Education':['Education_Graduate','Education_Not_Graduate'],
        'Self_Employed':['Self_Employed_No','Self_Employed_Yes'],
        'Gender':['Gender_Female','Gender_Male'],
        'Married':['Married_Yes','Married_No']}
    input_area =['Gender', 'Married', 'Dependents', 'Education',
        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area']
    txt_item=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term',]
    ctx={}
    res=0
#    sel_item=['Gender', 'Married', 'Dependents', 'Education','Credit_History', 'Property_Area']
    if request.POST:
       val={}
       print('your model ' , g_model)
       for item in input_area:
            val[item]= int(request.POST[item])
       for item in val.values():
           print('your inputs  ' ,item , type(item))
       res=modify(val,suff_list)

       #res=predictor(val)
       #return HttpResponse (res)

    ctx={'txt_item':txt_item ,'res':res}
    return TemplateResponse( request ,'file.html',ctx)
def modify(val ,suff_list):
    dummy_val ={
    'ApplicantIncome':0,
    'CoapplicantIncome':0,
    'LoanAmount':0,
    'Loan_Amount_Term':0,
    'Credit_History':0,
    'LoanAmount_log':0,
    'Gender_Female':0,
    'Gender_Male':0,
    'Married_No':0,
    'Married_Yes':0,
    'Dependents_3':0,
    'Dependents_0':0,
    'Dependents_1':0,
    'Dependents_2':0,
    'Education_Graduate':0,
    'Education_Not_Graduate':0,
    'Self_Employed_No':0,
    'Self_Employed_Yes':0,
    'Property_Area_Rural':0,
    'Property_Area_Semiurban':0,
    'Property_Area_Urban':0}

    for item in val.keys():
        if item in suff_list.keys():
            dummy_val[suff_list[item][val[item]]]=1
        else:
            dummy_val[item]=val[item]
    dummy_val['LoanAmount_log'] = np.log(dummy_val['LoanAmount'])
    print(val)
    print(dummy_val)
    return predict_ans(dummy_val)
def predict_ans(dummy_val):
    res_arr=g_model.predict(pd.DataFrame(dummy_val,index=['0']))
    res1 = bool(res_arr)
    return res1
