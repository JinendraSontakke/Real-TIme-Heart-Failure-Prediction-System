#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import category_encoders as encoders
from sklearn.preprocessing import RobustScaler
from sklearn import model_selection, metrics, naive_bayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC 
from lightgbm import LGBMClassifier
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# #### Import the Data

# In[2]:


df_heart = pd.read_csv('heart.csv')
df_heart.shape


# In[3]:


df_heart.isnull().sum()


# In[5]:


df_heart[numerical].describe()


# In[6]:


scaler = RobustScaler()
encoder_num = scaler.fit_transform(df_heart[numerical])
encoded_num = pd.DataFrame(encoder_num, columns =['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak'])
encoded_num.shape
print(encoded_num.head(10))


# In[7]:


# First, we get the target label

df_target = df_heart['HeartDisease']
df_target.columns = ['target']
df_target.value_counts()


# In[8]:



CATBoostENCODE = encoders.CatBoostEncoder()

encoder_cat = CATBoostENCODE.fit_transform(df_heart[categorical], df_target)
encoded_cat = pd.DataFrame(encoder_cat)


# In[9]:


encoded_cat.describe()


# #### Model Preparation

# In[10]:


# Prepare the training data set
df_train = df_heart.copy()
df_train.info()


# In[11]:


df_train.drop(numerical, axis=1, inplace=True)
df_train.drop(categorical, axis=1, inplace=True)
df_train = pd.concat([df_train, encoded_num, encoded_cat], axis=1) 
df_train.head(10)


# In[12]:


y = df_train.iloc[:,0:1]
X = df_train.iloc[:,1:]


# In[13]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)


# In[14]:


df_performance = pd.DataFrame(columns=['Model', 'Balanced Accuracy', 'Accuracy', 'Precision', 'F1', 'Recall', 'ROC AUC'])
df_performance


# In[15]:


def model_performance (p_test, p_train, p_test_prob, p_train_prob, Y_test, y_train, model_name):
    global df_performance
    predicted_test = pd.DataFrame(p_test)
    predicted_train = pd.DataFrame(p_train)
    print('=============================================')
    print('Scoring Metrics for {} (Validation)'.format(model_name))
    print('=============================================')
    print('Balanced Accuracy Score = {:2.3f}'.format(metrics.balanced_accuracy_score(Y_test, predicted_test)))
    print('Accuracy Score = {:2.3f}'.format(metrics.accuracy_score(Y_test, predicted_test)))
    print('Precision Score = {:2.3f}'.format(metrics.precision_score(Y_test, predicted_test)))
    print('F1 Score = {:2.3f}'.format(metrics.f1_score(Y_test, predicted_test, labels=['0','1'])))
    print('Recall Score = {:2.3f}'.format(metrics.recall_score(Y_test, predicted_test, labels=['0','1'])))
    print('ROC AUC Score = {:2.3f}'.format(metrics.roc_auc_score(Y_test, predicted_test, labels=['0','1'])))
    print('Confusion Matrix')
    print('==================')
    print(metrics.confusion_matrix(Y_test, predicted_test))
    print('==================')
    print(metrics.classification_report(Y_test, predicted_test, target_names=['0','1']))
    metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(Y_test, predicted_test)).plot()

    df_performance = df_performance.append({'Model':model_name
                                            , 'Balanced Accuracy': metrics.balanced_accuracy_score(Y_test, predicted_test)
                                            , 'Accuracy' :metrics.accuracy_score(Y_test, predicted_test)
                                            , 'Precision' :metrics.precision_score(Y_test, predicted_test)
                                            , 'F1':metrics.f1_score(Y_test, predicted_test, labels=['0','1'])
                                            , 'Recall': metrics.recall_score(Y_test, predicted_test, labels=['0','1'])
                                            , 'ROC AUC': metrics.roc_auc_score(Y_test, predicted_test, labels=['0','1'])
                                           }, ignore_index = True)

    
    fpr_test, tpr_test, _ = metrics.roc_curve(Y_test, p_test_prob)

    roc_auc_test = metrics.roc_auc_score(Y_test, predicted_test, labels=['0','1'])

    # Precision x Recall Curve
    precision_test, recall_test, thresholds_test = metrics.precision_recall_curve(Y_test, p_test_prob)

    print('=============================================')
    print('Scoring Metrics for {} (Training)'.format(model_name))
    print('=============================================')
    print('Balanced Accuracy Score = {:2.3f}'.format(metrics.balanced_accuracy_score(y_train, predicted_train)))
    print('Accuracy Score = {:2.3f}'.format(metrics.accuracy_score(y_train, predicted_train)))
    print('Precision Score = {:2.3f}'.format(metrics.precision_score(y_train, predicted_train)))
    print('F1 Score = {:2.3f}'.format(metrics.f1_score(y_train, predicted_train)))
    print('Recall Score = {:2.3f}'.format(metrics.recall_score(y_train, predicted_train, labels=['0','1'])))
    print('ROC AUC Score = {:2.3f}'.format(metrics.roc_auc_score(y_train, predicted_train, labels=['0','1'])))
    print('Confusion Matrix')
    print('==================')
    print(metrics.confusion_matrix(y_train, predicted_train))
    print('==================')
    print(metrics.classification_report(y_train, predicted_train, target_names=['0','1']))
    metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_train, predicted_train)).plot()

    fpr_train, tpr_train, _ = metrics.roc_curve(y_train, p_train_prob)

    roc_auc_train = metrics.roc_auc_score(y_train, predicted_train, labels=['0','1'])

    # Subplot of 1 x 2 matrix 
    print('======= ROC Curve and Precision x Recall Curve =======')
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].plot(fpr_test, tpr_test, color='darkorange', label='ROC curve - Validation (area = %0.3f)' % roc_auc_test)
    ax[0].plot(fpr_train, tpr_train, color='darkblue', label='ROC curve - Training (area = %0.3f)' % roc_auc_train)
    ax[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('ROC Curve')
    ax[0].legend(loc="lower right")

    # Precision x Recall Curve
    precision_train, recall_train, thresholds_train = metrics.precision_recall_curve(y_train, p_train_prob)
    ax[1].plot(recall_test, precision_test, color='darkorange')
    ax[1].plot(recall_train, precision_train, color='darkblue')
    ax[1].set_title('Precision-Recall Curve')
    ax[1].set_ylabel('Precision')
    ax[1].set_xlabel('Recall')

    plt.show()


# ### Model 1 - Regression

# In[19]:


p_cv = 5
p_score = 'accuracy'


# In[20]:


reg_param_grid = {
    'penalty': ['l1', 'l2'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}


# In[21]:


print('Train X = {}, Train Y ={}'.format(X_train.shape, y_train.shape))


# In[22]:


# clf_1 = LogisticRegression(max_iter=10000, class_weight='balanced', penalty='l2', C=1.0, solver='lbfgs')
clf_1 = LogisticRegression(max_iter=10000, C=0.1)
cv = model_selection.StratifiedKFold(n_splits=p_cv, random_state=100, shuffle=True)
model_1 = model_selection.GridSearchCV(clf_1, reg_param_grid, cv=cv, scoring=p_score, n_jobs=-1, verbose=1)
model_1.fit(X_train, y_train.values.ravel())


# In[23]:


print(model_1.best_estimator_)
print(model_1.best_params_)


# In[24]:


p_train_1 = model_1.predict(X_train)
p_test_1 = model_1.predict(X_test)
p_train_proba_1 = model_1.predict_proba(X_train)[:,1]
p_test_proba_1 = model_1.predict_proba(X_test)[:,1]


# In[25]:


model_performance(p_test_1, p_train_1, p_test_proba_1, p_train_proba_1, y_test, y_train, 'Logistic Regression')


# #### Model 2 - Decision Tree

# In[26]:


estimators = [25,50,75,100]
max_depth = [5]
min_samples_split = [20, 50, 75, 100, 150, 200]
min_samples_leaf = [25, 50, 75, 100, 150, 200]


# In[27]:


clf_2 = DecisionTreeClassifier()

tree_param_grid = { 
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf
}

cv = model_selection.StratifiedKFold(n_splits=p_cv, random_state=100, shuffle=True)

model_2 = model_selection.GridSearchCV(clf_2, tree_param_grid, cv=cv, scoring=p_score, n_jobs=-1, verbose=1)
model_2.fit(X_train, y_train)


# In[28]:


print(model_2.best_estimator_)
print(model_2.best_params_)


# In[29]:


p_train_2 = model_2.predict(X_train)
p_test_2 = model_2.predict(X_test)
p_train_proba_2 = model_1.predict_proba(X_train)[:,1]
p_test_proba_2 = model_1.predict_proba(X_test)[:,1]


# In[30]:


model_performance(p_test_2, p_train_2, p_test_proba_2, p_train_proba_2, y_test, y_train, 'Decision Tree')


# #### Model 3 - KNN

# In[31]:


model_3 = KNeighborsClassifier(n_neighbors=10)


# In[32]:


model_3.fit(X_train, y_train.values.ravel())


# In[33]:


p_train_3 = model_3.predict(X_train)
p_test_3 = model_3.predict(X_test)
p_train_proba_3 = model_3.predict_proba(X_train)[:,1]
p_test_proba_3 = model_3.predict_proba(X_test)[:,1]


# In[34]:


model_performance(p_test_3, p_train_3, p_test_proba_3, p_train_proba_3, y_test, y_train, 'kNN')


# #### Model 4 - Random Forest

# In[35]:


estimators = [25,50,75,100]
max_depth = [5]
min_samples_split = [20, 50, 75, 100, 150, 200]
min_samples_leaf = [25, 50, 75, 100, 150, 200]## Decision Tree


# In[36]:


clf_4 = RandomForestClassifier()

forest_params_grid={'n_estimators':estimators,
           'max_depth':max_depth,
           'min_samples_split':min_samples_split,
           'min_samples_leaf':min_samples_leaf  }


# In[37]:


cv = model_selection.StratifiedKFold(n_splits=p_cv, random_state=100, shuffle=True)

model_4 = model_selection.GridSearchCV(clf_4, forest_params_grid, cv=cv, scoring=p_score, n_jobs=-1, verbose=1)
model_4.fit(X_train, y_train.values.ravel())


# In[38]:


print(model_4.best_params_)
print(model_4.best_estimator_)


# In[39]:


p_train_4 = model_4.predict(X_train)
p_test_4 = model_4.predict(X_test)
p_train_proba_4 = model_4.predict_proba(X_train)[:,1]
p_test_proba_4 = model_4.predict_proba(X_test)[:,1]


# In[40]:


model_performance(p_test_4, p_train_4, p_test_proba_4, p_train_proba_4, y_test, y_train, 'Random Forest')


# ####  Model 5 - SVM Classifer

# In[42]:


C = [0.1, 1.0, 10, 100]
gamma = [0.1,0.01,0.001, 0.0001]
kernel = ['linear','rbf']


# In[43]:


clf_5 = SVC(probability=True)

SVC_params_grid={'C':C,
           'gamma':gamma,
           'kernel':kernel}


# In[44]:


cv = model_selection.StratifiedKFold(n_splits=p_cv, random_state=100, shuffle=True)

model_5 = model_selection.GridSearchCV(clf_5, SVC_params_grid, cv=cv, scoring=p_score, n_jobs=-1, verbose=1)
model_5.fit(X_train, y_train.values.ravel())


# In[45]:


print(model_5.best_params_)
print(model_5.best_estimator_)


# In[46]:


p_train_5 = model_5.predict(X_train)
p_test_5 = model_5.predict(X_test)
p_train_proba_5 = model_5.predict_proba(X_train)[:,1]
p_test_proba_5 = model_5.predict_proba(X_test)[:,1]


# In[47]:


model_performance(p_test_5, p_train_5, p_test_proba_5, p_train_proba_5, y_test, y_train, 'SVM')


# #### Model 6 - XGBoost

# In[48]:


model_6 = XGBClassifier(n_estimators=100, learning_rate=0.01, max_depth = 3, colsample_bytree = 0.8, subsample= 0.8, n_jobs=-1)


# In[49]:


model_6.fit(X_train, y_train.values.ravel())


# In[50]:


p_train_6 = model_6.predict(X_train)
p_test_6 = model_6.predict(X_test)
p_train_proba_6 = model_6.predict_proba(X_train)[:,1]
p_test_proba_6 = model_6.predict_proba(X_test)[:,1]


# In[51]:


model_performance(p_test_6, p_train_6, p_test_proba_6, p_train_proba_6, y_test, y_train, 'XGBoost')


# ####  Model 7 - LGBM

# In[52]:


model_7 = LGBMClassifier(random_state=1234, boosting_type= 'gbdt', objective= 'binary', feature_fraction=0.7, bagging_fraction=0.7, learning_rate=0.01, max_depth=1, silent=True, metric='auc', n_estimators=5000, n_jobs = -1)


# In[53]:


model_7.fit(X_train, y_train.values.ravel())


# In[54]:


p_train_7 = model_7.predict(X_train)
p_test_7 = model_7.predict(X_test)
p_train_proba_7 = model_7.predict_proba(X_train)[:,1]
p_test_proba_7 = model_7.predict_proba(X_test)[:,1]


# In[55]:


model_performance(p_test_7, p_train_7, p_test_proba_7, p_train_proba_7, y_test, y_train, 'LGBM')


# ####  Model 8 - Voting Ensemble

# In[56]:


# Instantiate the learners (classifiers)
learner_1 = LogisticRegression(**model_1.best_params_, max_iter=10000)
learner_2 = DecisionTreeClassifier(**model_2.best_params_)
learner_3 = KNeighborsClassifier(n_neighbors=10)
learner_4 = RandomForestClassifier(**model_4.best_params_)
learner_5 = XGBClassifier(n_estimators=100, learning_rate=0.01, max_depth = 3, colsample_bytree = 0.8, subsample= 0.8, n_jobs=-1)
learner_6 = naive_bayes.GaussianNB()
learner_7 = SVC(gamma=0.001, probability=True)
learner_8 = LGBMClassifier(random_state=1234, boosting_type= 'gbdt', objective= 'binary', feature_fraction=0.7, bagging_fraction=0.7, learning_rate=0.01, max_depth=1, silent=True, metric='auc', n_estimators=5000, n_jobs = -1)


# ## Hard Voting 

# In[57]:


# Instantiate the voting classifier
hard_voting = VotingClassifier([('LogReg', learner_1),
                           ('Tree', learner_2),
                           ('KNN', learner_3),
                           ('Forest', learner_4),
                          ('XGBoost', learner_5),
                          ('NB', learner_6),
                          ('LGBM', learner_8)
                               ],
                            voting='hard')


# In[58]:


hard_voting.fit(X_train, y_train.values.ravel())


# In[59]:


p_train_hardvoting = hard_voting.predict(X_train)
p_test_hardvoting = hard_voting.predict(X_test)


# In[60]:


predicted_test = pd.DataFrame(p_test_hardvoting)
predicted_train = pd.DataFrame(p_train_hardvoting)
print('=============================================')
print('Scoring Metrics for Hard Voting (Validation)')
print('=============================================')
print('Balanced Accuracy Score = {}'.format(metrics.balanced_accuracy_score(y_test, predicted_test)))
print('Accuracy Score = {}'.format(metrics.accuracy_score(y_test, predicted_test)))
print('Precision Score = {}'.format(metrics.precision_score(y_test, predicted_test)))
print('F1 Score = {}'.format(metrics.f1_score(y_test, predicted_test, labels=['0','1'])))
print('Recall Score = {}'.format(metrics.recall_score(y_test, predicted_test, labels=['0','1'])))
print('ROC AUC Score = {}'.format(metrics.roc_auc_score(y_test, predicted_test, labels=['0','1'])))
print('Confusion Matrix')
print('==================')
print(metrics.confusion_matrix(y_test, predicted_test))
print('==================')
print(metrics.classification_report(y_test, predicted_test, target_names=['0','1']))
metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_test, predicted_test)).plot()

df_performance = df_performance.append({'Model':'Hard Voting'
                                        , 'Balanced Accuracy': metrics.balanced_accuracy_score(y_test, predicted_test)
                                        , 'Accuracy' :metrics.accuracy_score(y_test, predicted_test)
                                        , 'Precision' :metrics.precision_score(y_test, predicted_test)
                                        , 'F1':metrics.f1_score(y_test, predicted_test, labels=['0','1'])
                                        , 'Recall': metrics.recall_score(y_test, predicted_test, labels=['0','1'])
                                        , 'ROC AUC': metrics.roc_auc_score(y_test, predicted_test, labels=['0','1'])
                                       }, ignore_index = True)


print('=============================================')
print('Scoring Metrics for Hard Voting (Training)')
print('=============================================')
print('Balanced Accuracy Score = {}'.format(metrics.balanced_accuracy_score(y_train, predicted_train)))
print('Accuracy Score = {}'.format(metrics.accuracy_score(y_train, predicted_train)))
print('Precision Score = {}'.format(metrics.precision_score(y_train, predicted_train)))
print('F1 Score = {}'.format(metrics.f1_score(y_train, predicted_train)))
print('Recall Score = {}'.format(metrics.recall_score(y_train, predicted_train, labels=['0','1'])))
print('ROC AUC Score = {}'.format(metrics.roc_auc_score(y_train, predicted_train, labels=['0','1'])))
print('Confusion Matrix')
print('==================')
print(metrics.confusion_matrix(y_train, predicted_train))
print('==================')
print(metrics.classification_report(y_train, predicted_train, target_names=['0','1']))
metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_train, predicted_train)).plot()


# ## Soft Voting

# In[61]:


# Instantiate the voting classifier
soft_voting = VotingClassifier([('LogReg', learner_1),
                           ('Tree', learner_2),
                           ('KNN', learner_3),
                           ('Forest', learner_4),
                          ('XGBoost', learner_5),
                          ('NB', learner_6),
                          ('LGBM', learner_8)
                               ],
                            voting='soft')


# In[62]:


soft_voting.fit(X_train, y_train.values.ravel())


# In[63]:


p_train_softvoting = soft_voting.predict(X_train)
p_test_softvoting = soft_voting.predict(X_test)
p_train_proba_softvoting = soft_voting.predict_proba(X_train)[:,1]
p_test_proba_softvoting = soft_voting.predict_proba(X_test)[:,1]


# In[64]:


model_performance(p_test_softvoting, p_train_softvoting, p_test_proba_softvoting, p_train_proba_softvoting, y_test, y_train, 'Soft Voting')


# In[68]:


df_performance = df_performance.round(2)
df_performance


# In[ ]:




