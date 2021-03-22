import numpy as np
import pandas as pd

import copy 
import os
import json

import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.datasets import make_classification
from plotly.subplots import make_subplots

def plot_roc(y, y_score):
    fpr, tpr, thresholds = roc_curve(y, y_score)

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.show()
    
def plot_pr_curve(y, y_score):    
    precision, recall, thresholds = precision_recall_curve(y, y_score)

    fig = px.area(
        x=recall, y=precision,
        title=f'Precision-Recall Curve (AUC={auc(recall, precision):.4f})',
        labels=dict(x='Recall', y='Precision'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=1, y1=0
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.show()    
    
def plot_roc_train_test(y_train, y_score_train, y_test, y_score_test):
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    Y_Scores = [y_score_train, y_score_test]
    Y_Obs = [y_train, y_test]
    Y_Names = ["Train", "Test"]
    for i in range(len(Y_Scores)):
        y_true = Y_Obs[i]
        y_score = Y_Scores[i]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)

        name = f"{Y_Names[i]} ROC (AUC={auc_score:.4f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500
    )
    fig.show()
    
def plot_pr_train_test(y_train, y_score_train, y_test, y_score_test):
    #fig = go.Figure()
    fig = make_subplots(rows=1, cols=2)
    
    # subplot for ROC curve
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1,
        row=1, col=1
    )
    Y_Scores = [y_score_train, y_score_test]
    Y_Obs = [y_train, y_test]
    Y_Names = ["Train", "Test"]
    for i in range(len(Y_Scores)):
        y_true = Y_Obs[i]
        y_score = Y_Scores[i]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)

        name = f"{Y_Names[i]} ROC (AUC={auc_score:.4f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'),row=1, col=1)
    
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
    
    # subplot for PR curve
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=1, y1=0, 
        row=1,col=2
    )
    Y_Scores = [y_score_train, y_score_test]
    Y_Obs = [y_train, y_test]
    Y_Names = ["Train", "Test"]
    for i in range(len(Y_Scores)):
        y_true = Y_Obs[i]
        y_score = Y_Scores[i]

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auc_score = average_precision_score(y_true, y_score)
    
        name = f"{Y_Names[i]} PR (AUC={auc_score:.4f})"
        fig.add_trace(go.Scatter(x=recall, y=precision, name=name, mode='lines'),row=1,col=2)

    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    fig.update_layout(
        title="ROC & Precision-Recall Curves",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=1000, height=500
    )
    fig.show()
    
def hyper_table(path='structured_data_classifier'):
    trial_json = os.popen('ls ./{}/trial_*/trial.json'.format(path)).read().split('\n')[:-1]
    DATA = []
    for file in trial_json:
        with open(file) as f: 
            DATA.append(json.load(f))
    for k in range(len(DATA)):
        DATA[k]['hyperparameters']['values']['score'] = DATA[k]['score']
    hyper_df = pd.concat([pd.DataFrame.from_dict(data['hyperparameters']['values'], orient='index') for data in DATA], axis=1)
    hyper_df.columns = ["trial#{}".format(k+1) for k in range(len(DATA))]
    return(hyper_df)

def predic_error_analysis(x_train, y_train, y_score_train, x_test, y_test, y_score):
    df1 = copy.deepcopy(x_train)
    df1['obs'] = y_train
    df1['predict'] = y_score_train
    df1['data'] = 'Train'
    #display(df1)
    df2 = copy.deepcopy(x_test)
    df2['obs'] = y_test
    df2['predict'] = y_score
    df2['data'] = 'Test'
    #display(df2)
    df3 = pd.concat([df1,df2])

    fig = px.scatter(
        df3, x='obs', y='predict',
        marginal_x='histogram', marginal_y='histogram',
        color='data', trendline='ols'
    )
    fig.update_traces(histnorm='probability', selector={'type':'histogram'})
    fig.add_shape(
        type="line", line=dict(dash='dash'),
        x0=df3['obs'].min(), y0=df3['obs'].min(),
        x1=df3['obs'].max(), y1=df3['obs'].max()
    )
    fig.update_layout(title="Prediction Error Analysis", 
                      yaxis=dict(range=[df3['obs'].min(), df3['obs'].max()]),
                      xaxis=dict(range=[df3['obs'].min(), df3['obs'].max()]),
                      width=1000, height=1000)
    fig.show()
    

# ROC PR curves for multi-class
def plot_pr_multi_class(y_train, y_score_train, y_test, y_score_test):

    y_class = np.unique(y_train)
    class_dict_test = {}
    for c in y_class:
        score = [s[c] for s in y_score_test]
        obs = y_test == c
        class_dict_test[c] = {'obs': obs, 'score': score}

    class_dict_train = {}
    for c in y_class:
        score = [s[c] for s in y_score_train]
        obs = y_train == c
        class_dict_train[c] = {'obs': obs, 'score': score}

    
    fig = make_subplots(rows=2, cols=2)
    
    # subplot for ROC curve: train data
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1,
        row=1, col=1)
    
    Y_Scores = [class_dict_train[c]['score'] for c in class_dict_train]
    Y_Obs = [class_dict_train[c]['obs'] for c in class_dict_train]
    Y_Names = y_class
    for i in range(len(Y_Scores)):
        y_true = Y_Obs[i]
        y_score = Y_Scores[i]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)

        name = f"class {Y_Names[i]} ROC: Train (AUC={auc_score:.4f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'),row=1, col=1)
    
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate (Train)", row=1, col=1)

    # subplot for ROC curve: train data
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1,
        row=1, col=2)
    
    Y_Scores = [class_dict_test[c]['score'] for c in class_dict_test]
    Y_Obs = [class_dict_test[c]['obs'] for c in class_dict_test]
    Y_Names = y_class
    for i in range(len(Y_Scores)):
        y_true = Y_Obs[i]
        y_score = Y_Scores[i]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)

        name = f"class {Y_Names[i]} ROC: Test (AUC={auc_score:.4f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'),row=1, col=2)
    
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
    fig.update_yaxes(title_text="True Positive Rate (Test)", row=1, col=2)

    # subplot for PR curve: train data
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=1, y1=0, 
        row=2,col=1)
    Y_Scores = [class_dict_train[c]['score'] for c in class_dict_train]
    Y_Obs = [class_dict_train[c]['obs'] for c in class_dict_train]
    Y_Names = y_class
    for i in range(len(Y_Scores)):
        y_true = Y_Obs[i]
        y_score = Y_Scores[i]

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auc_score = average_precision_score(y_true, y_score)
    
        name = f"class {Y_Names[i]} PR Train (AUC={auc_score:.4f})"
        fig.add_trace(go.Scatter(x=recall, y=precision, name=name, mode='lines'),row=2,col=1)

    fig.update_xaxes(title_text="Recall", row=2, col=1)
    fig.update_yaxes(title_text="Precision (Train)", row=2, col=1)

    # subplot for PR curve: test data
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=1, y1=0, 
        row=2,col=2)
    Y_Scores = [class_dict_test[c]['score'] for c in class_dict_test]
    Y_Obs = [class_dict_test[c]['obs'] for c in class_dict_test]
    Y_Names = y_class
    for i in range(len(Y_Scores)):
        y_true = Y_Obs[i]
        y_score = Y_Scores[i]

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auc_score = average_precision_score(y_true, y_score)
    
        name = f"class {Y_Names[i]} PR Test (AUC={auc_score:.4f})"
        fig.add_trace(go.Scatter(x=recall, y=precision, name=name, mode='lines'),row=2,col=2)

    fig.update_xaxes(title_text="Recall", row=2, col=2)
    fig.update_yaxes(title_text="Precision (Test)", row=2, col=2)

    fig.update_layout(
        title="ROC & Precision-Recall Curves",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=1200, height=1000)
    fig.show()