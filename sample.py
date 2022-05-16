# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
#### 1. Linear Regression
# (https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)


# 전처리 샘플
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/My Drive/2021 생물정보학실습/1. Linear regression/linear_regression_data.csv')
df




# 시각화
# !pip install adjustText # 텍스트 위치 조정해주는 라이브러리


x = np.log(df['Cumulative number of divisions of all stem cells per lifetime (lscd)'].to_numpy())
y = np.log(df['Lifetime cancer risk'].to_numpy())
plt.figure(figsize=(20, 10)) 

plt.figure(figsize=(20, 10)) 
plt.title('The relationship between the number of stem cell divisions in the lifetime of a given tissue and the lifetime risk of cancer in that tissue')
plt.xlabel('Total stem cell divisions')
plt.ylabel('Lifetime risk')
plt.scatter(x, y) # x와 y의 개수가 같아야 함.

text = []
for xpoint, ypoint, name in zip(x,y,df['Cancer type']):
    text.append(plt.text(xpoint, ypoint, name))
adjust_text(text) # 텍스트


plt.show() # 보여주라~






# 학습_통째로
model = LinearRegression()              # 모델 생성
result = model.fit(x.reshape(-1,1), y)  # 현재 가진 x,y로 학습시킨 결과를 저장. result가 객체겟지?

print('coef_:', result.coef_) # 계수
print('intercept_:', result.intercept_) # 절편

result.predict(x.reshape(-1, 1))


# 학습_분리 및 오차 계산

from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state=42)

model = LinearRegression()
result = model.fit(xTrain.reshape(-1,1), yTrain.reshape(-1,1))

from sklearn.metrics import mean_squared_error
print('Training loss: ', mean_squared_error(  yTrain, result.predict(xTrain.reshape(-1,1)) ) )
print('Test loss: ', mean_squared_error( yTest ,result.predict(xTest.reshape(-1,1))) )









# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
#### 2. Logistic Regression (https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV # logistic regression. CV는 cross validation 에 사용.
from sklearn.model_selection import train_test_split # 샘플 나누는 작업
from sklearn.model_selection import GridSearchCV # 그리드서치. 최적의 파라미터(penalty 등등) 찾아줌.

xTrain, xTest, yTrain, yTest = train_test_split(np.array(lung['mutations']).reshape(-1,1), lung['label'], test_size = 0.1) # 9:1로 나눔.


grid_values = { # 어떤 형태의 학습을 사용할지(lr이 logistic regression object니까 그대로 최적화됨.)
              'penalty': ['l1','l2', 'elasticnet', None],
              'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
              'cv':[2,3,4,5,6]
               }

lr = LogisticRegressionCV() # cross validation 할 수 있는 객체 생성
model_lr = GridSearchCV(lr, param_grid=grid_values) 
model_lr.fit(xTrain, yTrain) # 학습

model_lr.score(xTest, yTest)




# confusion matrix
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(model_lr, xTest, yTest) 





# ROC
from sklearn.metrics import plot_roc_curve # roc_curve 그려주는 lib.
plot_roc_curve(model_lr, xTest, yTest) # 학습한 모델 객체 / xtest / ytest값 넣어주면 알아서 계산해 보여줌.

# 추가적으로 이 내용들을 작성해도 됨.
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--') # 대각선 선 하나 그려줌.
plt.xlim([0.0, 1.0]) # x축 범위
plt.ylim([0.0, 1.05]) # y축 범위
plt.xlabel('False Positive Rate') # x축이름
plt.ylabel('True Positive Rate') # y축이름
plt.title('Receiver operating characteristic example') # 이름
plt.legend(loc="lower right") # 레전드 어디다 둘지
plt.show() # 보여줘보쇼







# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
#### 3. Random Forest (https://scikit-learn.org/stable/modules/tree.html / https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)


classifier를 쓸 수도 있고, regressor를 쓸 수도 있다. 

# 정보 보기
!pip install missingno
import missingno as msno

msno.matrix(df)

# 학습하기
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']

# Maximum number of levels in tree
max_depth = list(range(1,30))
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# 연속형 범주나 많은 범주를 가진 feature들을 과대평가하는 경향이 있기 때문에 이를 완화시키는 옵션
# mean_impurity_decrease = 0 # float 형태

class_weight = ['balanced', 'balanced_subsample']

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'class_weight': class_weight}
# 파라미터 참고 : (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html?highlight=randomizedsearchcv#sklearn-model-selection-randomizedsearchcv)




from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2)


rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, 
                               param_distributions = random_grid, 
                               n_iter = 100, 
                               cv = 3, 
                               verbose=2, 
                               random_state=42, 
                               n_jobs = 4)
rf_random.fit(xTrain, yTrain)


rf_random.cv_results_
rf_random.best_estimator_
rf_random.best_score_
rf_random.best_params_


# 최고의 모델에 대한 성능평가
best_model = rf_random.best_estimator_

best_model.feature_importances_ # 피쳐 중요성 평가
feature_df = pd.DataFrame(best_model.feature_importances_, columns = ['Value'])
feature_df.plot(kind='barh', title = 'Feature importance')


best_model.score(xTest, yTest) # 점수평가





# confusion matrix
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(best_model, xTest, yTest) 





# ROC
from sklearn.metrics import plot_roc_curve # roc_curve 그려주는 lib.
plot_roc_curve(best_model, xTest, yTest)

plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')



# XGboost 는 ensemble 모델의 한 종류로 boosting을 해줌.
쓸 일 없을 거 같은데.. random_forest_11_11.ipynb 마지막 부분 참고.





# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
#### 4. Neural Network
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html# 참고. classification, regression 등등.

import os
import pickle
import innvestigate
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import Model
from keras.optimizers import Adam
from keras.models import load_model
from gen_matrix import GenMatrix
from parse_file import ParseFile
from load_data import LoadData
from analysis import Analyse
from keras_rand_nn import RandNN, RunRandNN


from main import *
#import tensorflow.keras as keras



train_input_df, input_num, num_feature, rna_df, cpnum_df, num_gene, num_pathway = LoadData(dir_opt).pre_load_train()
matrixA = GenMatrix.feature_gene_matrix(num_feature, num_gene) # 피쳐
matrixB = GenMatrix.gene_pathway_matrix() # 정답

print('RUNING DEEP NERUAL NETWORK...')
# BUILD NEURAL NETWORK
print('BUILDING CUSTOMED NERUAL NETWORK...')
layer0 = num_pathway # 46개 input
layer1 = 256 # 중간중간들
layer2 = 128
layer3 = 32
epoch = 5 
batch_size = 256
verbose = 0
learning_rate = 0.001
end_epoch = 5

input_model, gene_model, pathway_model, model = build_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3)


이거 말고 그냥 sklearn documentation에서 제공하는 대로 하자.. 엡바임;






# feature selection
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.metrics import mean_squared_error
from itertools import product

# loss = []

# for mode, k_select in product((chi2, f_classif), (20,40,60,80,100)):
#     print(mode, k_select)
#     X_train_new = SelectKBest(mode, k=k_select).fit_transform(rna, y)  # feature selection
#     # 테스트를 위해서 전체 데이터에서 먼저 feature selection을 한 후 train_test_split을 함
    
#     X_train, X_test, y_train, y_test = train_test_split(X_train_new, y, test_size=0.15, random_state=42)
    
#     # train
#     grid_feature_selected.fit(X_train, y_train)
#     forest = grid_feature_selected.best_estimator_ # select best estimator

#     y_pred = forest.predict(X_test)
#     mse_loss = mean_squared_error(y_test, y_pred) # loss 계산 / regression이므로 mse로 로스 비교해서 모델 선택
#     # classification 문제라면 f1, auc 등등 계산 가능
    
#     loss.append(mse_loss)


from sklearn.feature_selection import SelectKBest, f_classif, chi2
xtrain_new = SelectKBest(chi2, k=20).fit_transform(xTrain, yTrain) # 20개의 중요한 feature를 추출.


# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
#### 5. Clustering

https://scikit-learn.org/stable/modules/clustering.html

















msno로 결측치 체크 및 버리기 등.
dropna()



