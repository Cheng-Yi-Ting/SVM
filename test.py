from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest  # 得到k個最高分的feature
from sklearn.feature_selection import chi2  # 卡方驗證，用於分類
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import classification_report, accuracy_score

# 使用datasets 20類新聞中取4個類別
categories = ['alt.atheism', 'talk.religion.misc',
              'comp.graphics', 'sci.space']
data_train = fetch_20newsgroups(subset='train', categories=categories)
data_test = fetch_20newsgroups(subset='test', categories=categories)


# 分詞,去除停用詞,並且建立VSM模型
vectorizer = CountVectorizer(
    stop_words='english', min_df=2)  # 設定停用詞，min_df用來降維
X = vectorizer.fit_transform(data_train.data)  # X是稀疏矩陣

# 計算特徵與結果的相依性
ch2 = SelectKBest(chi2, k=500)  # 選擇卡方值最大的前500個特徵
X_train = ch2.fit_transform(X, data_train.target)  # 得到新的訓練集

# 建立test的VSM，不用fit_transform，改用transform可以讓訓練和測試有相同的文檔集合
x_test = vectorizer.transform(data_test.data)
# 測試集的卡方驗證也選出和訓練集一樣的特徵
X_test = ch2.transform(x_test)
# print(X_test)
# print(X_train)

clf = SVC()
param_grid = {'kernel': ('linear', 'poly'), 'C': [0.1, 1]}

grid_search = GridSearchCV(clf, param_grid=param_grid,
                           cv=5, scoring=make_scorer(accuracy_score))
grid_search.fit(X_train, data_train.target)
y = grid_search.predict(X_test)
print(grid_search.best_estimator_)

print(accuracy_score(data_test.target, y))
print(classification_report(data_test.target, y))
