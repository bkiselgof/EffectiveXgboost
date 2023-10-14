##################################################################################################################################
### Chapter 2.
##################################################################################################################################

import pandas as pd

import urllib.request
import zipfile




url = 'https://github.com/mattharrison/datasets/raw/master/data/kaggle-survey-2018.zip'
fname = 'kaggle-survey-2018.zip'
member_name = 'multipleChoiceResponses.csv'

def extract_zip(src, dst, member_name):
    url = src
    fname = dst
    fin = urllib.request.urlopen(url)
    data = fin.read()
    with open(dst, mode='wb') as fout:
        fout.write(data)
    with zipfile.ZipFile(dst) as z:
        kag = pd.read_csv(z.open(member_name))
        kag_questions = kag.iloc[0]
        raw = kag.iloc[1:]
        return raw


def extract_zip2(src, dst, member_name):
    url = src
    fname = dst
    #fin = urllib.request.urlopen(url)
    #data = fin.read()
    #with open(dst, mode='wb') as fout:
        #fout.write(data)
    with zipfile.ZipFile(dst) as z:
        kag = pd.read_csv(z.open(member_name))
        kag_questions = kag.iloc[0]
        raw = kag.iloc[1:]
        return raw

raw = extract_zip(url, fname, member_name)

raw = extract_zip2(url, fname, member_name)
   

def topn(ser, n=5, default='other'):
    counts = ser.value_counts()
    return ser.where(ser.isin(counts.index[:n]), default)


def tweak_kag(df_: pd.DataFrame) -> pd.DataFrame:
    return (df_
           .assign(age=df_.Q2.str.slice(0,2).astype(int),
                       education=df_.Q4.replace({'Master’s degree': 18,
                         'Bachelor’s degree': 16,
                         'Doctoral degree': 20,
                         'Some college/university study without earning a bachelor’s degree': 13,
                         'Professional degree': 19,
                         'I prefer not to answer': None,
                         'No formal education past high school': 12}),
                       major=(df_.Q5
                              .pipe(topn, n=3)
                              .replace({
                        'Computer science (software engineering, etc.)': 'cs',
                        'Engineering (non-computer focused)': 'eng',
                        'Mathematics or statistics': 'stat'})),
                       years_exp=(df_.Q8.str.replace('+','', regex=False)
                           .str.split('-', expand=True)
                           .iloc[:,0]
                           .astype(float)),
                       compensation=(df_.Q9.str.replace('+','', regex=False)
                           .str.replace(',','', regex=False)
                           .str.replace('500000', '500', regex=False)
                           .str.replace('I do not wish to disclose my approximate yearly compensation', '0', regex=False)
                           .str.split('-', expand=True)
                           .iloc[:,0]
                           .fillna(0)
                           .astype(int))              
                       .loc[:, 'Q1,Q3,age,education,major,years_exp,compensation,''python,r,sql'.split(',')]
                           ))


from feature_engine import encoding, imputation
from sklearn import base, pipeline

class TweakKagTransformer(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, ycol=None):
        self.ycol = ycol
        
    def transform(self, X):
        return tweak_kag(X)
    
    def fit(self, X, y=None):
        return self
    

def get_rawX_y(df, y_col):
    raw = (df.query('Q3.isin(["United States of America", "China", "India"]) '
                   'and Q6.isin(["Data Scientist", "Software Engineer"])'))
    return raw.drop(columns=[y_col]), raw[y_col]

kag_pl = pipeline.Pipeline(
    [('tweak', TweakKagTransformer()),
    ('cat', encoding.OneHotEncoder(top_categories=5, drop_last=True, 
                                    variables=['Q1', 'Q3', 'major'])),
    ('num_impute', imputation.MeanMedianImputer(imputation_method='median',
                                    variables=['education', 'years_exp']))]
     )

raw.head()
raw.info()

raw.columns
(raw.query('Q3.isin(["United States of America", "China", "India"]) '
                   'and Q6.isin(["Data Scientist", "Software Engineer"])'))


from sklearn import model_selection
kag_X, kag_y = get_rawX_y(raw, 'Q6')

kag_X_train, kag_X_test, kag_y_train, kag_y_test = model_selection.train_test_split(
    kag_X, kag_y, test_size=.3, stratify=kag_y, random_state=42)

X_train = kag_pl.fit_transform(kag_X_train, kag_y_train)
X_test = kag_pl.transform(kag_X_test)
print(X_train)

kag_y_train


##################################################################################################################################
### Chapter 3 - EDA.
##################################################################################################################################


(X_train
 .assign(data_scientist = kag_y_train == 'Data Scientist')
 .corr(method='spearman')
 .style
 .background_gradient(cmap='RdBu', vmax=1, vmin=1)
 .set_sticky(axis='index')
 )

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8,4))
(X_train
 .assign(data_scientist = kag_y_train)
 .groupby('r')
 .data_scientist
 .value_counts()
 .unstack()
 .plot.bar(ax=ax)
 )

fig, ax = plt.subplots(figsize=(8,4))
(pd.crosstab(index=X_train['major_cs'], columns=kag_y)
 .plot.bar(ax=ax)
  )

fig, ax = plt.subplots(figsize=(8,4))
(X_train
 .plot.scatter(x='years_exp', y='compensation', alpha=.3, ax=ax, c='purple')
 )

import seaborn.objects as so

fig, ax = plt.subplots(figsize=(8,4))
(so
 .Plot(X_train.assign(title=kag_y_train), x='years_exp', y='compensation', color='title')
 .add(so.Dots(alpha=.3, pointsize=2), so.Jitter(x=.5, y=10_000))
 .add(so.Line(), so.PolyFit())
 .on(fig)
 .plot()
 )


fig, ax = plt.subplots(figsize=(8,4))
(so
 .Plot(X_train.
       assign(    
        title=kag_y_train,
        country=(X_train
                 .loc[:, 'Q3_United States of America': 'Q3_China']
                 .idxmax(axis='columns')
                 )
                 ), x='years_exp', y='compensation', color='title')
 .facet('country')
 .add(so.Dots(alpha=.01, pointsize=2, color='gray'), so.Jitter(x=.5, y=10_000), col=None)
 .add(so.Dots(alpha=.5, pointsize=1.5, color='gray'), so.Jitter(x=.5, y=10_000))
 .add(so.Line(pointsize=1), so.PolyFit(order=2))
 .scale(x=so.Continuous().tick(at=[0,1,2,3,4,5]))
 .limit(y=(-10_000, 200_000), x=(-1,6))
 .on(fig)
 .plot()
 )


##################################################################################################################################
### Chapter 4 - Tree Creation
##################################################################################################################################


import numpy as np
import numpy.random as rn

pos_center = 12
pos_count = 100
neg_center = 7
neg_count = 1000
rs = rn.RandomState(rn.MT19937(rn.SeedSequence(42)))
gini = pd.DataFrame({'value':
        np.append((pos_center) + rs.randn(pos_count),
                (neg_center) + rs.randn(neg_count)),
                    'label':
                ['pos']*pos_count + ['neg']*neg_count})


fig, ax = plt.subplots(figsize=(8,4))
_ = (gini
     .groupby('label')
     [['value']]
     .plot.hist(bins=30, alpha=.5, edgecolor='black')
    )
ax.legend(['Negative', 'Positive'])


##################################################################################################################################
### Chapter 5 - Stumps on Real Data
##################################################################################################################################

from sklearn import tree

stump_dt = tree.DecisionTreeClassifier(max_depth=1)
X_train = kag_pl.fit_transform(kag_X_train)
stump_dt.fit(X_train, kag_y_train)

fig, ax = plt.subplots(figsize=(8,4))
features = list(c for c in X_train.columns)
tree.plot_tree(stump_dt, feature_names = features, filled=True, class_names=stump_dt.classes_, ax=ax)

X_test = kag_pl.transform(kag_X_test)
stump_dt.score(X_test, kag_y_test)

from sklearn import dummy
dummy_model = dummy.DummyClassifier()
dummy_model.fit(X_train, kag_y_train)
dummy_model.score(X_test, kag_y_test)

import xgboost as xgb
kag_stump = xgb.XGBClassifier(n_estimators=1, max_depth=1)
kag_stump.fit(X_train, kag_y_train)

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(kag_y_train)
y_test = label_encoder.fit_transform(kag_y_test)
y_test[:5]

label_encoder.classes_
label_encoder.inverse_transform([0,1])

kag_stump = xgb.XGBClassifier(n_estimators=1, max_depth=1)
kag_stump.fit(X_train, y_train)
kag_stump.score(X_test, y_test)


my_dot_export(kag_stump, num_trees=0, filename='kag/stump_xg_kag.dot', title='XGBoost Stump')


##################################################################################################################################
### Chapter 6 - Hyperparameters
##################################################################################################################################

from sklearn import tree
import matplotlib.pyplot as plt

# Underfitting
underfit = tree.DecisionTreeClassifier(max_depth=1)
X_train = kag_pl.fit_transform(kag_X_train)
underfit.fit(X_train, kag_y_train)
underfit.score(X_test, kag_y_test)

# Overfitting with Decitions Trees
hi_variance = tree.DecisionTreeClassifier(max_depth=None)
X_train = kag_pl.fit_transform(kag_X_train)
hi_variance.fit(X_train, kag_y_train)
hi_variance.score(X_test, kag_y_test)


fig, ax = plt.subplots(figsize=(8,4))
features = list(c for c in X_train.columns)
tree.plot_tree(hi_variance, feature_names=features, filled=True)
                   

features = list(c for c in X_train.columns)
_ = tree.plot_tree(hi_variance, feature_names=features, filled=True, 
                   class_names=['SE', 'DS'])


# limit view to first 2
fig, ax = plt.subplots(figsize=(8,4))
features = list(c for c in X_train.columns)
tree.plot_tree(hi_variance, feature_names=features, filled=True, 
                   class_names=hi_variance.classes_, max_depth=2, fontsize=6)

##################################################################################################################################
### Chapter 7 - Tree Hyperparameters
##################################################################################################################################

from sklearn import tree

stump_dt = tree.DecisionTreeClassifier(max_depth=1)
X_train = kag_pl.fit_transform(kag_X_train)
stump_dt.fit(X_train, kag_y_train)
stump_dt.get_params()

# Validation curves

accuracies = []
for depth in range(1,15):
    between = tree.DecisionTreeClassifier(max_depth=depth)
    between.fit(X_train, kag_y_train)
    accuracies.append(between.score(X_test, kag_y_test))
fig, ax = plt.subplots(figsize=(10,4))
(pd.Series(accuracies, name='Accuracy', index=range(1,len(accuracies)+1))
 .plot(ax=ax, title='Accuracy at a given Tree Depth'))
ax.set_ylabel('Accuracy')
ax.set_xlabel('max_depth')
 
# Check the graph
between = tree.DecisionTreeClassifier(max_depth=7)
between.fit(X_train, kag_y_train)
between.score(X_test, kag_y_test)

# Yellowbrick package
from yellowbrick.model_selection import validation_curve
fig, ax = plt.subplots(figsize=(10,4))
viz = validation_curve(tree.DecisionTreeClassifier(),
    X=pd.concat([X_train, X_test]),
    y=pd.concat([kag_y_train, kag_y_test]),
    param_name='max_depth', param_range=range(1,14),
    scoring='accuracy', cv=5, ax=ax, n_jobs=6)

# Grid Search
from sklearn.model_selection import GridSearchCV
params = {
    'max_depth': [3,5,7,8],
    'min_samples_leaf': [1,3,4,5,6],
    'min_samples_split': [2,3,4,5,6],
}
grid_search = GridSearchCV(estimator=tree.DecisionTreeClassifier(),
                           param_grid=params, cv=4, n_jobs=1,
                           verbose=1, scoring='accuracy')
grid_search.fit(pd.concat([X_train, X_test]),
                pd.concat([kag_y_train, kag_y_test]))

grid_search.best_params_

between2 = tree.DecisionTreeClassifier(**grid_search.best_params_)
between2.fit(X_train, kag_y_train)
between2.score(X_test, kag_y_test)

(pd.DataFrame(grid_search.cv_results_)
 .sort_values(by='rank_test_score')
 .style
 .background_gradient(axis='rows')
 )


# Validate grid search results
results = model_selection.cross_val_score(
    tree.DecisionTreeClassifier(max_depth=7),
    X=pd.concat([X_train, X_test], axis='index'),
    y=pd.concat([kag_y_train, kag_y_test], axis='index'),
    cv=4
)

results
results.mean()

results = model_selection.cross_val_score(
    tree.DecisionTreeClassifier(max_depth=7, min_samples_leaf=5,
                                min_samples_split=2),
    X=pd.concat([X_train, X_test], axis='index'),
    y=pd.concat([kag_y_train, kag_y_test], axis='index'),
    cv=4
)

results
results.mean()

##################################################################################################################################
### Chapter 8 - Random Forest
##################################################################################################################################
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import ensemble
import numpy as np

# Transform the data for XGBoost
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(kag_y_train)
y_test = label_encoder.fit_transform(kag_y_test)
y_test[:5]
###############################################

rf = ensemble.RandomForestClassifier(random_state=42)
rf.fit(X_train, kag_y_train)
rf.score(X_test, kag_y_test)

rf.get_params()
len(rf.estimators_)
rf.estimators_[0]

fig, ax = plt.subplots(figsize=(10,4))
features = list(c for c in X_train.columns)
tree.plot_tree(rf.estimators_[0], feature_names=features, filled=True, 
                   class_names=rf.classes_, ax=ax, max_depth=2, fontsize=6)


# XGBoost Random Forest
import xgboost as xgb
rf_xg = xgb.XGBRFClassifier(random_state=42)
rf_xg.fit(X_train, y_train)
rf_xg.score(X_test, y_test)

rf_xg.get_params()

fig, ax = plt.subplots(figsize=(6,12))
xgb.plot_tree(rf_xg, num_trees=0, ax=ax, size='1,1')

# Yellowbrick package
from yellowbrick.model_selection import validation_curve
fig, ax = plt.subplots(figsize=(10,4))
viz = validation_curve(xgb.XGBRFClassifier(random_state=42),
    X=pd.concat([X_train, X_test], axis='index'),
    y=np.concatenate([y_train, y_test]),
    param_name='n_estimators', param_range=range(1,100, 2),
    scoring='accuracy', cv=3, ax=ax)

# XGBoost with optimal number of estimators from the graph = 29
import xgboost as xgb
rf_xg29 = xgb.XGBRFClassifier(random_state=42, n_estimators=29)
rf_xg29.fit(X_train, y_train)
rf_xg29.score(X_test, y_test)

##################################################################################################################################
### Chapter 9 - XGBoost
##################################################################################################################################

#import dtreeviz
from feature_engine import encoding, imputation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import base, compose, datasets, ensemble, metrics, model_selection, pipeline, preprocessing, tree
import scikitplot
import xgboost as xgb
import yellowbrick.model_selection as ms
from yellowbrick import classifier
import urllib
import zipfile
import xg_helpers as xhelp


# Testing regular xgboost model on data from previous chapters. Can't install xg_helpers package
import xgboost as xgb
xg_oob = xgb.XGBRFClassifier()
xg_oob.fit(X_train, y_train)
xg_oob.score(X_test, y_test)

# Try depth 2 and 2 trees
xg2 = xgb.XGBRFClassifier(max_depth=2, n_estimators=2)
xg2.fit(X_train, y_train)
xg2.score(X_test, y_test)

xgb.plot_tree(xg2, num_trees=0)


def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))

inv_logit(-0.08476+0.0902701)

##################################################################################################################################
### Chapter 10 - Early Stopping
##################################################################################################################################

# Defaults
xg = xgb.XGBClassifier()
xg.fit(X_train, y_train)
xg.score(X_test, y_test)

# Early Stopping
xg = xgb.XGBClassifier(early_stopping_rounds=20)
xg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
xg.score(X_test, y_test)

xg.best_ntree_limit

results = xg.evals_result()
results

results = xg.evals_result()
fig, ax = plt.subplots(figsize=(8,4))
ax = (pd.DataFrame({'training': results['validation_0']['logloss'],
              'testing': results['validation_1']['logloss'],
             })
.assign(ntrees=lambda adf: range(1,len(adf)+1))
.set_index('ntrees')
.plot(figsize=(5,4), ax=ax, title='eval_results with early stopping'))
ax.annotate('Best number \nof trees (13)', xy=(13,.498), xytest=(20,.42), arrowprops={'color':'k'})
ax.set_xlabel('ntrees')

# Optimal number of trees = 13
xg13 = xgb.XGBClassifier(n_estimators=13)
xg13.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
xg13.score(X_test, y_test)


# Changing error metrics
xg_err = xgb.XGBClassifier(early_stopping_rounds=20, eval_metric='error')
xg_err.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
xg_err.score(X_test, y_test)

# Print optimal number of trees                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
xg_err.best_ntree_limit


##################################################################################################################################
### Chapter 11 - XGBoost Hyperparameters
##################################################################################################################################

xg = xgb.XGBClassifier()
xg.fit(X_train, y_train)
xg.get_params()


from yellowbrick.model_selection import validation_curve
fig, ax = plt.subplots(figsize=(8,4))
ms.validation_curve(xgb.XGBRFClassifier(), X_train, y_train,
    param_name='gamma', param_range=[0,.5,1,5,10,20,30],
    n_jobs=-1, ax=ax)

# Testing the impact on learning rates
xg_lr1 = xgb.XGBClassifier(learning_rate=1, max_depth=2)
xg_lr1.fit(X_train, y_train)
xg_lr1.score(X_test, y_test)

xg_lr001 = xgb.XGBClassifier(learning_rate=.001, max_depth=2)
xg_lr001.fit(X_train, y_train)
xg_lr001.score(X_test, y_test)


from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
params = {
    'reg_lambda': [0],
    'learning_rate': [.1,.3],
    'subsample': [.7,1],
    'max_depth': [2,3],
    'random_state': [42],
    'n_jobs': [-1],
    'n_estimators': [200],
}

xgb2 = xgb.XGBClassifier(early_stopping_rounds=5)

cv = (model_selection.GridSearchCV(xgb2,params, cv=4, n_jobs=1).
        fit(X_train, y_train, eval_set=[(X_test, y_test)],
            verbose=50
        ))
                           
cv.best_params_

xgb_grid = xgb.XGBClassifier(**cv.best_params_, early_stopping_rounds=50)
xgb_grid.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_test, y_test)],
            verbose=10)
xgb_grid.score(X_test, y_test)

# Default model
xgb_def = xgb.XGBClassifier(early_stopping_rounds=50)
xgb_def.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_test, y_test)],
            verbose=10)
xgb_def.score(X_test, y_test)

xgb_grid.score(X_test, y_test), xgb_def.score(X_test, y_test)


X = pd.concat([X_train, X_test], axis='index')
y = pd.Series([*y_train, *y_test], index=X.index)


results_default = model_selection.cross_val_score(
    xgb.XGBClassifier(),
    X=X, y=y,
    cv=4
)

results_default
results_default.mean()

params = cv.best_params_

results_grid = model_selection.cross_val_score(
    xgb.XGBClassifier(**params),
    X=X, y=y,
    cv=4
)

results_grid
results_grid.mean()

##################################################################################################################################
### Chapter 12 - Hyperopt
##################################################################################################################################


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score, roc_auc_score  
from typing import Any, Dict, Union


def hyperparameter_tuning(space: Dict[str, Union[float, int]],
                    X_train:pd.DataFrame, y_train: pd.Series,
                    X_test:pd.DataFrame, y_test: pd.Series,
                    early_stopping_rounds: int=50,
                    metric:callable=accuracy_score) -> Dict[str, Any]:
    int_vals = ['max_depth', 'reg_alpha']
    space = {k: (int(val) if k in int_vals else val)
                for k, val in space.items()}
    space['early_stopping_rounds'] = early_stopping_rounds
    model = xgb.XGBClassifier(**space) 
    evaluation = [(X_train, y_train),
            (X_test, y_test)]
    model.fit(X_train, y_train,
                 eval_set=evaluation, verbose=False)    
    pred = model.predict(X_test)
    score = metric(y_test, pred)
    return {'loss': -score, 'status': STATUS_OK, 'model': model}


#https://bradleyboehmke.github.io/xgboost_databricks_tuning/index.html#slide21
options = {
    'learning_rate': hp.loguniform('learning_rate', -7, 0),
    'max_depth': hp.quniform('max_depth', 1, 8, 1),
    'min_child_weight': hp.loguniform('min_child_weight', -2, 3),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'gamma': hp.loguniform('gamma', -10, 10),
    'reg_alpha': hp.loguniform('alpha', 0, 10),
    'reg_lambda': hp.loguniform('lambda', 1, 10),
    'random_state': 42 
    }

trials = Trials()
best = fmin(fn=lambda space: hyperparameter_tuning(space, X_train, y_train, X_test, y_test),
    space=options,           
    algo=tpe.suggest,            
    max_evals=2000,            
    trials=trials,
    timeout=60*5 # 5 minutes
           )
print (best)

# best params from the book
long_params ={'colsample_bytree': 0.6874845219014455,
 'gamma': 0.06936323554883501,
 'learning_rate': 0.214392142849,
 'max_depth': 6, 
 'min_child_weight': 0.6678357091609912,
 'reg_alpha': 3.2979862933185546,
 'reg_lambda': 7.850943400390477,
 'subsample': 0.999767483950891}

xgb_ex = xgb.XGBClassifier(**long_params, early_stopping_rounds=50, 
                                          n_estimators=50)
xgb_ex.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_test, y_test)],
            verbose=10)
xgb_ex.score(X_test, y_test)


from hyperopt import hp, pyll
pyll.stochastic.sample(hp.choice('value', ['a', 'b', 'c']))

# pull a number from uniform distribution
pyll.stochastic.sample(hp.uniform('value', 0,1))

# pull a number from uniform distribution 10,000 samples
uniform_vals = [pyll.stochastic.sample(hp.uniform('value', 0,1)) for _ in range(10_000)]

fig, ax = plt.subplots(figsize=(8,4))
ax.hist(uniform_vals)
plt.show()


# loguniform distributuion
loguniform_vals = [pyll.stochastic.sample(hp.loguniform('value', -5,5)) for _ in range(10_000)]

fig, ax = plt.subplots(figsize=(8,4))
ax.hist(loguniform_vals)
plt.show()

# Another graph
fig, ax = plt.subplots(figsize=(8,4))
(pd.Series(np.arange(-5,5, step=.1))
.rename('x')
.to_frame()
.assign(y=lambda adf:np.exp(adf.x))
.plot(x='x', y='y', ax=ax)
)

from typing import Any, Dict, Sequence
def trial2df(trial: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    vals=[]
    for t in trial:
        result = t['result']
        misc = t['misc']
        val = {k:(v[0] if isinstance(v,list) else v)
               for k,v in misc['vals'].items()}
        val['loss'] = result['loss']
        val['tid'] = t['tid']
        vals.append(val)
    return pd.DataFrame(vals)

hyper2hr = trial2df(trials)

import seaborn as sns
fig, ax = plt.subplots(figsize=(8,4))
sns.heatmap(hyper2hr.corr(method='spearman'), cmap='RdBu', annot=True, fmt='.2f', vmin=-1, vmax=1, ax=ax)

fig, ax = plt.subplots(figsize=(8,4))
(hyper2hr.plot.scatter(x='tid', y='loss', alpha=.1, color='purple', ax=ax))

fig, ax = plt.subplots(figsize=(8,4))
(hyper2hr.plot.scatter(x='max_depth', y='loss', alpha=1, color='purple', ax=ax))

import numpy as np
def jitter(df, col, amount: float=1) -> pd.Series:
    vals = np.random.uniform(low=-amount/2, high=amount/2, size=df.shape[0])
    return df[col] + vals

fig, ax = plt.subplots(figsize=(8,4))
(hyper2hr
 .assign(max_depth=lambda df:jitter(df, 'max_depth', amount=.8))
 .plot.scatter(x='max_depth', y='loss', alpha=1, color='purple', ax=ax))


fig, ax = plt.subplots(figsize=(8,4))
(hyper2hr
 .assign(max_depth=lambda df:jitter(df, 'max_depth', amount=.8))
 .plot.scatter(x='max_depth', y='loss', alpha=.5, cmap='viridis', ax=ax))

# violin plot
fig, ax = plt.subplots(figsize=(8,4))
sns.violinplot(x='max_depth', y='loss', data=hyper2hr, kind='violin', ax=ax)


fig, ax = plt.subplots(figsize=(8,4))
(hyper2hr
 .plot.scatter(x='gamma', y='colsample_bytree', alpha=.8, color='purple', cmap='viridis', ax=ax))


##################################################################################################################################
### Chapter 13 - Step-wise Tunining with Hyperopt
##################################################################################################################################


from hyperopt import fmin, tpe, hp, Trials
params = {'random_state': 42}
rounds = [{'max_depth': hp.quniform('max_depth', 1, 8, 1),
           'min_child_weight': hp.loguniform('min_child_weight', -2, 3)},
           {'subsample': hp.uniform('subsample', 0.5, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1)},
            {'reg_alpha': hp.loguniform('reg_alpha', 0, 10),
            'reg_lambda': hp.loguniform('lambda', 1, 10)},
            {'gamma': hp.loguniform('gamma', -10, 10),},
            {'learning_rate': hp.loguniform('learning_rate', -7, 0)}]

all_trials = []
for round in rounds:
    params = {**params, **round}
    trials = Trials()
    best = fmin(fn=lambda space: hyperparameter_tuning(space, X_train, y_train, X_test, y_test),
    space=params,           
    algo=tpe.suggest,            
    max_evals=40,            
    trials=trials,
          )
    params = {**params, **best}
    all_trials.append(trials)

best

# Model with step parameters from above
xg_step = xgb.XGBClassifier(**best, early_stopping_rounds=50, 
                                          n_estimators=500)
xg_step.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_test, y_test)],
            verbose=100)
xg_step.score(X_test, y_test)

# best parameters from the book
step_params ={'colsample_bytree': 0.6235721099295888,
 'gamma': 0.00011273797329538491,
 'learning_rate': 0.24399020050740935,
 'max_depth': 5, 
 'min_child_weight': 0.6411044640540848,
 'random_state': 42,
 'subsample': 0.9492383155577023}

xg_step1 = xgb.XGBClassifier(**step_params, early_stopping_rounds=50, 
                                          n_estimators=500)
xg_step1.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_test, y_test)],
            verbose=100)
xg_step1.score(X_test, y_test)

##################################################################################################################################
### Chapter 14 - Do you have enough data
##################################################################################################################################


X = pd.concat([X_train, X_test], axis='index')
y = pd.Series([*y_train, *y_test], index=X.index)


params = {
    'reg_lambda': 0,
    'learning_rate': 0.3,
    'subsample': 1,
    'max_depth': 2,
    'random_state': 42,
    'n_jobs': -1,
    'n_estimators': 200
}

# Regular learning curve
fig, ax = plt.subplots(figsize=(8,4))
import yellowbrick.model_selection as ms
viz = ms.learning_curve(xgb.XGBClassifier(**params), 
                        X, y, ax=ax
                        )
ax.set_ylim(0.6,1)

# Ok model
fig, ax = plt.subplots(figsize=(8,4))
import yellowbrick.model_selection as ms
viz = ms.learning_curve(xgb.XGBClassifier(max_depth=7), 
                        X, y, ax=ax
                        )
viz.ax.set_ylim(0.6,1)


# Underfitting model
fig, ax = plt.subplots(figsize=(8,4))
import yellowbrick.model_selection as ms
viz = ms.learning_curve(tree.DecisionTreeClassifier(max_depth=1), 
                        X, y, ax=ax
                        )
ax.set_ylim(0.6,1)

# Overfitting model
fig, ax = plt.subplots(figsize=(8,4))
import yellowbrick.model_selection as ms
viz = ms.learning_curve(tree.DecisionTreeClassifier(), 
                        X, y, ax=ax
                        )
ax.set_ylim(0.6,1)

##################################################################################################################################
### Chapter 15 - Model evaluation
##################################################################################################################################


xgb_def = xgb.XGBClassifier()
xgb_def.fit(X_train, y_train)
xgb_def.score(X_test, y_test)

# Same as default score
from sklearn import metrics
metrics.accuracy_score(y_test, xgb_def.predict(X_test))

fig, ax = plt.subplots(figsize=(8,4))
classifier.confusion_matrix(xgb_def, X_train, y_train,
                            X_test, y_test,
                            classes=['DS', 'SE'], ax=ax
                           )

# Another way
cm = metrics.confusion_matrix(y_test, xgb_def.predict(X_test))
cm

fig, ax = plt.subplots(figsize=(8,4))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=['DS', 'SE'])
disp.plot(ax=ax, cmap='Blues')

# Using fractions instead of numbers
cm = metrics.confusion_matrix(y_test, xgb_def.predict(X_test), normalize='true')
fig, ax = plt.subplots(figsize=(8,4))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=['DS', 'SE'])
disp.plot(ax=ax, cmap='Blues')

# Precision and recall

metrics.precision_score(y_test, xgb_def.predict(X_test))

metrics.recall_score(y_test, xgb_def.predict(X_test))

# PR curve
from yellowbrick import classifier
fig, ax = plt.subplots(figsize=(8,4))
classifier.precision_recall_curve(xgb_def, X_train, y_train, X_test, y_test, 
                                 micro=False, macro=False, ax=ax, per_class=True
                        )
ax.set_ylim(0, 1.05)

# F1 score
metrics.f1_score(y_test, xgb_def.predict(X_test))

print(metrics.classification_report(y_test, y_pred=xgb_def.predict(X_test),
                              target_names=['DS', 'SE']))

# ROC curve
fig, ax = plt.subplots(figsize=(8,8))
metrics.RocCurveDisplay.from_estimator(xgb_def,
                                    X_test, y_test, ax=ax, label='default')
metrics.RocCurveDisplay.from_estimator(xg_step,
                                    X_test, y_test, ax=ax)

##################################################################################################################################
### Chapter 16 - Training for different metrics
##################################################################################################################################


from yellowbrick import model_selection as ms
fig, ax = plt.subplots(figsize=(8,4))
ms.validation_curve(xgb.XGBClassifier(), X_train, y_train,
            scoring='accuracy', param_name='learning_rate', 
            param_range=[0.001, 0.01, .05, .1, .2, .5, .9, .1],
            ax=ax)
ax.set_xlabel('Accuracy')

fig, ax = plt.subplots(figsize=(8,4))
ms.validation_curve(xgb.XGBClassifier(), X_train, y_train,
            scoring='roc_auc', param_name='learning_rate', 
            param_range=[0.001, 0.01, .05, .1, .2, .5, .9, .1],
            ax=ax)
ax.set_xlabel('roc_auc')


# Step-wise recall tuning
from hyperopt import fmin, tpe, hp, Trials
from sklearn.metrics import roc_auc_score

params = {'random_state': 42}

rounds = [{'max_depth': hp.quniform('max_depth', 1, 9, 1),
           'min_child_weight': hp.loguniform('min_child_weight', -2, 3)},
           {'subsample': hp.uniform('subsample', 0.5, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1)},
            {'gamma': hp.loguniform('gamma', -10, 10),},
            {'learning_rate': hp.loguniform('learning_rate', -7, 0)}]

for round in rounds:
    params = {**params, **round}
    trials = Trials()
    best = fmin(fn=lambda space: hyperparameter_tuning(space, X_train, y_train, X_test, y_test, metric=roc_auc_score),
    space=params,           
    algo=tpe.suggest,            
    max_evals=40,            
    trials=trials,
          )
    params = {**params, **best}
    

best
params
step_params = params

step_params['max_depth'] = 6

# XGBoost default
xgb_def = xgb.XGBClassifier()
xgb_def.fit(X_train, y_train)
metrics.roc_auc_score(y_test, xgb_def.predict(X_test))


# best parameters from the book
step_params ={'colsample_bytree': 0.6235721099295888,
 'gamma': 0.00011273797329538491,
 'learning_rate': 0.24399020050740935,
 'max_depth': 5, 
 'min_child_weight': 0.6411044640540848,
 'random_state': 42,
 'subsample': 0.9492383155577023}

xgb_tuned = xgb.XGBClassifier(**step_params, early_stopping_rounds=50, 
                                          n_estimators=500)
xgb_tuned.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_test, y_test)],
            verbose=100)
metrics.roc_auc_score(y_test, xgb_tuned.predict(X_test))

##################################################################################################################################
### Chapter 17 - Model Interpretation
##################################################################################################################################

# Logistic regression
from sklearn import linear_model, preprocessing
std = preprocessing.StandardScaler()
lr = linear_model.LogisticRegression(penalty=None)
lr.fit(std.fit_transform(X_train), y_train)
lr.score(std.transform(X_test), y_test)

lr.coef_

fig, ax = plt.subplots(figsize=(8,4))
(pd.Series(lr.coef_[0], index=X_train.columns)
 .sort_values()
 .plot.barh(ax=ax)
 )

# Decision Treee
tree7 = tree.DecisionTreeClassifier(max_depth=7)
tree7.fit(X_train, y_train)
tree7.score(X_test, y_test)

fig, ax = plt.subplots(figsize=(8,4))
(pd.Series(tree7.feature_importances_, index=X_train.columns)
 .sort_values()
 .plot.barh(ax=ax)
 )

# XGBoost feature importances

xgb_def = xgb.XGBClassifier()
xgb_def.fit(X_train, y_train)
xgb_def.score(X_test, y_test)

fig, ax = plt.subplots(figsize=(8,4))
(pd.Series(xgb_def.feature_importances_, index=X_train.columns)
 .sort_values()
 .plot.barh(ax=ax)
 )

fig, ax = plt.subplots(figsize=(8,4))
xgb.plot_importance(xgb_def, importance_type='cover', ax=ax)

fig, ax = plt.subplots(figsize=(8,4))
xgb.plot_importance(xgb_def, importance_type='gain', ax=ax)

fig, ax = plt.subplots(figsize=(8,4))
xgb.plot_importance(xgb_def, importance_type='weight', ax=ax)


##################################################################################################################################
### Chapter 18 - xgbfir
##################################################################################################################################

import xgbfir
xgbfir.saveXgbFI(xgb_def, feature_names=X_train.columns, OutputXlsxFile='fir.xlsx')

fir = pd.read_excel('fir.xlsx')
print(fir
      .sort_values(by='Average Rank')
      .head()
      .round(1)
      )

(X_train
 .assign(software_eng=y_train)
 .corr(method='spearman')
 .loc[:, ['education', 'years_exp', 'major_cs', 'r', 'compensation', 'age']]
 .style
 .background_gradient(cmap='RdBu', vmin=-1, vmax=1)
 .format('{:.2f}')
)

import seaborn.objects as so
fig, ax = plt.subplots(figsize=(8,4))
(so
 .Plot(X_train.assign(software_eng=y_train), x='years_exp', y='education', color='software_eng')
 .add(so.Dots(alpha=.9, pointsize=2), so.Jitter(x=.7, y=1))
 .add(so.Line(), so.PolyFit())
 .scale(color='viridis')
 .on(fig)
 .plot()
 )


(X_train
 .assign(software_eng=y_train)
 .groupby(['software_eng', 'r', 'major_cs'])
 .age
 .count()
 .unstack()
 .unstack()
 )



print(pd.read_excel('fir.xlsx', sheet_name='Interaction Depth 2').iloc[:20]
      .sort_values(by='Average Rank')
      .head(6)
        )


##################################################################################################################################
### Chapter 19 - SHAP
##################################################################################################################################

https://www.topbots.com/explainable-ai-marketing-analytics/

import shap


# best parameters from the book
step_params ={'colsample_bytree': 0.6235721099295888,
 'gamma': 0.00011273797329538491,
 'learning_rate': 0.24399020050740935,
 'max_depth': 5, 
 'min_child_weight': 0.6411044640540848,
 'random_state': 42,
 'subsample': 0.9492383155577023}

xg_step = xgb.XGBClassifier(**step_params, early_stopping_rounds=50, 
                                          n_estimators=500)
xg_step.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_test, y_test)],
            verbose=100)
xg_step.score(X_test, y_test)

import shap
shap.initjs()

# With jitter
fig, ax = plt.subplots(figsize=(8,4))
shap.plots.scatter(vals[:, 'education'], ax=ax, color = vals[:, 'years_exp'], x_jitter=1, alpha=.5)




# Heatmaps and correlations

# Correlations for features
import seaborn as sns
fig, ax = plt.subplots(figsize=(8,4))
sns.heatmap(X_test
            .assign(software_eng=y_test)
            .corr(method='spearman')
            .loc[:, ['age', 'education', 'years_exp', 'compensation',
                     'r', 'major_cs', 'software_eng']],
            cmap='RdBu', annot=True, fmt='.2f', vmin=-1, vmax=1, ax=ax)
plt.show()

# Correlations for shap values
import seaborn as sns
fig, ax = plt.subplots(figsize=(8,4))
sns.heatmap(shap_df
            .assign(software_eng=y_test)
            .corr(method='spearman')
            .loc[:, ['age', 'education', 'years_exp', 'compensation',
                     'r', 'major_cs', 'software_eng']],
            cmap='RdBu', annot=True, fmt='.2f', vmin=-1, vmax=1, ax=ax)
plt.show()

# Shap heatmap
shap.plots.heatmap(vals)



# Beesawrm graph
fig = plt.subplots(figsize=(8,4))
shap.plots.beeswarm(vals)

# Beesawrm graph
from matplotlib import cm
fig = plt.subplots(figsize=(8,4))
shap.plots.beeswarm(vals, max_display=len(X_test.columns), color=cm.autumn_r)

# Shap with no interactions

no_int_params = {'random_state': 42,
                 'max_depth': 1
                }

xg_no_int = xgb.XGBClassifier(**no_int_params, early_stopping_rounds=50, 
                                          n_estimators=500)
xg_no_int.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_test, y_test)],
            verbose=100)
xg_no_int.score(X_test, y_test)

shap_ind = shap.TreeExplainer(xg_no_int)
shap_ind_vals = shap_ind(X_test)

fig = plt.subplots(figsize=(8,4))
shap.plots.beeswarm(shap_ind_vals, max_display=len(X_test.columns))

fig, ax = plt.subplots(figsize=(8,4))
shap.plots.scatter(vals[:, 'years_exp'], ax=ax, color = vals[:, 'age'], x_jitter=1, alpha=.5)


fig, ax = plt.subplots(figsize=(8,4))
shap.plots.scatter(shap_ind_vals[:, 'years_exp'], ax=ax, color = vals[:, 'age'], x_jitter=1, alpha=.5)

##################################################################################################################################
### Chapter 19 - ICE, PDP plots
##################################################################################################################################


## Better Models with ICE, Partial Dependence, Monotonic Constraints, and Calibration

### ICE Plots

xgb_def = xgb.XGBClassifier(random_state=42)
xgb_def.fit(X_train, y_train)
xgb_def.score(X_test, y_test)

from sklearn.inspection import PartialDependenceDisplay
fig, axes = plt.subplots(ncols=2, figsize=(8,4))
PartialDependenceDisplay.from_estimator(xgb_def, X_train, features=['r', 'education'],
                                       kind='individual', ax=axes)

# To make it more clear, centered=True, makes graphs start from the same value
fig, axes = plt.subplots(ncols=2, figsize=(8,4))
PartialDependenceDisplay.from_estimator(xgb_def, X_train, features=['r', 'education'],
                                       centered=True,
                                       kind='individual', ax=axes)


fig, axes = plt.subplots(ncols=2, figsize=(8,4))
ax_h0 = axes[0].twinx()
ax_h0.hist(X_train.r, zorder=0)

ax_h1 = axes[1].twinx()
ax_h1.hist(X_train.education, zorder=0)

PartialDependenceDisplay.from_estimator(xgb_def, X_train, features=['r', 'education'],
                                        centered=True,
                                        ice_lines_kw={'zorder':10},
                                        kind='individual', ax=axes)
fig.tight_layout()


# Custom function
def quantile_ice(clf, X, col, center=True, q=10, color='k', alpha=.5, legend=True,
                add_hist=False, title='', val_limit=10, ax=None):
  """
    Generate an ICE plot for a binary classifier's predicted probabilities split 
    by quantiles.

    Parameters:
    ----------
    clf : binary classifier
        A binary classifier with a `predict_proba` method.
    X : DataFrame
        Feature matrix to predict on with shape (n_samples, n_features).
    col : str
        Name of column in `X` to plot against the quantiles of predicted probabilities.
    center : bool, default=True
        Whether to center the plot on 0.5.
    q : int, default=10
        Number of quantiles to split the predicted probabilities into.
    color : str or array-like, default='k'
        Color(s) of the lines in the plot.
    alpha : float, default=0.5
        Opacity of the lines in the plot.
    legend : bool, default=True
        Whether to show the plot legend.
    add_hist : bool, default=False
        Whether to add a histogram of the `col` variable to the plot.
    title : str, default=''
        Title of the plot.
    val_limit : num, default=10
        Maximum number of values to test for col.
    ax : Matplotlib Axis, deafault=None
        Axis to plot on.

    Returns:
    -------
    results : DataFrame
        A DataFrame with the same columns as `X`, as well as a `prob` column with 
        the predicted probabilities of `clf` for each row in `X`, and a `group` 
        column indicating which quantile group the row belongs to.
  """                  
  probs = clf.predict_proba(X)
  df = (X
        .assign(probs=probs[:,-1],
               p_bin=lambda df_:pd.qcut(df_.probs, q=q, 
                                        labels=[f'q{n}' for n in range(1,q+1)])
               )
       )
  groups = df.groupby('p_bin')

  vals = X.loc[:,col].unique()
  if len(vals) > val_limit:
    vals = np.linspace(min(vals), max(vals), num=val_limit)
  res = []
  for name,g in groups:
    for val in vals:
      this_X = g.loc[:,X.columns].assign(**{col:val})
      q_prob = clf.predict_proba(this_X)[:,-1]
      res.append(this_X.assign(prob=q_prob, group=name))
  results = pd.concat(res, axis='index')     
  if ax is None:
    fig, ax = plt.subplots(figsize=(8,4))
  if add_hist:
    back_ax = ax.twinx()
    back_ax.hist(X[col], density=True, alpha=.2) 
  for name, g in results.groupby('group'):
    g.groupby(col).prob.mean().plot(ax=ax, label=name, color=color, alpha=alpha)
  if legend:
    ax.legend()
  if title:
    ax.set_title(title)
  return results

fig, ax = plt.subplots(figsize=(8,4))
quantile_ice(xgb_def, X_train, 'education', q=10, legend=False, add_hist=True, ax=ax,
            title='ICE plot for Age')



### ICE Plots with SHAP


import shap

# Incorrect plot function from the book.
fig, ax = plt.subplots(figsize=(8,4))
shap.plots.partial_dependence_plot(ind='education', 
    model=lambda rows: xgb_def.predict_proba(rows)[:,-1],
    data=X_train.iloc[0:1000], ice=True, 
    npoints=(X_train.education.nunique()),
    pd_linewidth=0, show=False, ax=ax)
ax.set_title('ICE plot (from SHAP)')


# Corrected shap plot function.
fig, ax = plt.subplots(figsize=(8,4))
shap.plots.partial_dependence(ind='education', 
    model=lambda rows: xgb_def.predict_proba(rows)[:,-1],
    data=X_train.iloc[0:1000], ice=True, 
    npoints=(X_train.education.nunique()),
    pd_linewidth=0, show=False, ax=ax)
ax.set_title('ICE plot (from SHAP)')



### Partial Dependence Plots

fig, axes = plt.subplots(ncols=2, figsize=(8,4))
PartialDependenceDisplay.from_estimator(xgb_def, X_train, features=['r', 'education'],
                                        kind='average', ax=axes)
fig.tight_layout()

# PDP on top of ICE Plot
fig, axes = plt.subplots(ncols=2, figsize=(8,4))
PartialDependenceDisplay.from_estimator(xgb_def, X_train, features=['r', 'education'],
                                        centered=True, kind='both',
                                        ax=axes)
fig.tight_layout()


fig, axes = plt.subplots(ncols=2, figsize=(8,4))
PartialDependenceDisplay.from_estimator(xgb_def, X_train, features=['years_exp', 'Q1_Male'],
                                        centered=True, kind='both',
                                        ax=axes)
fig.tight_layout()


### PDP with SHAP

import shap
fig, ax = plt.subplots(figsize=(8,4))
col = 'years_exp'  
shap.plots.partial_dependence(ind=col,
                             model=lambda rows: xgb_def.predict_proba(rows)[:,-1],
                             data=X_train.iloc[0:1000], ice=False, 
                             npoints=(X_train[col].nunique()),
                             pd_linewidth=2, show=False, ax=ax)
ax.set_title('PDP plot (from SHAP)')


fig, ax = plt.subplots(figsize=(8,4))
col = 'years_exp'  
shap.plots.partial_dependence(ind=col, 
                             model=lambda rows: xgb_def.predict_proba(rows)[:,-1],
                             data=X_train.iloc[0:1000], ice=True, 
                             npoints=(X_train[col].nunique()),
                             model_expected_value=True,
                             feature_expected_value=True,
                             pd_linewidth=2, show=False, ax=ax)
ax.set_title('PDP plot (from SHAP) with ICE Plots')


### Monotonic Constraints

fig, ax = plt.subplots(figsize=(8,4))

(X_test
 .assign(target=y_test)
 .corr(method='spearman')
 .iloc[:-1]
 .loc[:,'target']
 .sort_values(key=np.abs)
 .plot.barh(title='Spearman Correlation with Target', ax=ax)
)

print(X_train
.assign(target=y_train)
.groupby('education')
.mean()
.loc[:, ['age', 'years_exp', 'target']]
)

X_train.education.value_counts()

print(raw
.query('Q3.isin(["United States of America", "China", "India"]) '
        'and Q6.isin(["Data Scientist", "Software Engineer"])') 
.query('Q4 == "Professional degree"')
.pipe(lambda df_:pd.crosstab(index=df_.Q5, columns=df_.Q6))
)

xgb_const = xgb.XGBClassifier(random_state=42,
          monotone_constraints={'years_exp':1, 'education':-1})
xgb_const.fit(X_train, y_train)
xgb_const.score(X_test, y_test)
```


small_cols = ['age', 'education', 'years_exp', 'compensation', 'python', 'r', 'sql',
              #'Q1_Male', 'Q1_Female', 'Q1_Prefer not to say',
              #'Q1_Prefer to self-describe', 
              'Q3_United States of America', 'Q3_India',
              'Q3_China', 'major_cs', 'major_other', 'major_eng', 'major_stat']
xgb_const2 = xgb.XGBClassifier(random_state=42,
          monotone_constraints={'years_exp':1, 'education':-1})
xgb_const2.fit(X_train[small_cols], y_train)

xgb_const2.score(X_test[small_cols], y_test)
#0.7569060773480663


fig, ax = plt.subplots(figsize=(8,4))
(pd.Series(xgb_def.feature_importances_, index=X_train.columns)
 .sort_values()
 .plot.barh(ax=ax)
)

fig, ax = plt.subplots(figsize=(8,4))
(pd.Series(xgb_const2.feature_importances_, index=small_cols)
 .sort_values()
 .plot.barh(ax=ax)
)


### Calibrating a Model

from sklearn.calibration import CalibratedClassifierCV

xgb_cal = CalibratedClassifierCV(xgb_def, method='sigmoid', cv='prefit')
xgb_cal.fit(X_test, y_test)

xgb_cal_iso = CalibratedClassifierCV(xgb_def, method='isotonic', cv='prefit')
xgb_cal_iso.fit(X_test, y_test)


### Calibration Curves

from sklearn.calibration import CalibrationDisplay
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(8,6))
gs = GridSpec(4, 3)
axes = fig.add_subplot(gs[:2, :3])
display = CalibrationDisplay.from_estimator(xgb_def, X_test, y_test, 
                                            n_bins=10, ax=axes)
disp_cal = CalibrationDisplay.from_estimator(xgb_cal, X_test, y_test, 
                                      n_bins=10,ax=axes, name='sigmoid')
disp_cal_iso = CalibrationDisplay.from_estimator(xgb_cal_iso, X_test, y_test, 
                                      n_bins=10, ax=axes, name='isotonic')
row = 2
col = 0
ax = fig.add_subplot(gs[row, col])
ax.hist(display.y_prob, range=(0,1), bins=20)
ax.set(title='Default', xlabel='Predicted Prob')
ax2 = fig.add_subplot(gs[row, 1])
ax2.hist(disp_cal.y_prob, range=(0,1), bins=20)
ax2.set(title='Sigmoid', xlabel='Predicted Prob')
ax3 = fig.add_subplot(gs[row, 2])
ax3.hist(disp_cal_iso.y_prob, range=(0,1), bins=20)
ax3.set(title='Isotonic', xlabel='Predicted Prob')
fig.tight_layout()

xgb_cal.score(X_test, y_test)
#0.7480662983425415

xgb_cal_iso.score(X_test, y_test)
#0.7491712707182321

xgb_def.score(X_test, y_test)
#0.7458563535911602

