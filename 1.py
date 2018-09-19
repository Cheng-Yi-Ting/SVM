from sklearn import svm

X = [[0, 0], [1, 1], [2, 1]]  # training samples
y = [0, 1, 4]  # training target
clf = svm.SVC()  # class
clf.fit(X, y)  # training the svc model

result = clf.predict([[2, 3]])  # predict the target of testing samples
print(result)  # target
