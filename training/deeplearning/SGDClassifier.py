from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

sgd = SGDClassifier(loss='log', max_iter=100, tol=1e-3, random_state=42)
cancer = load_breast_cancer()

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size=0.2, random_state=42)

sgd.fit(x_train, y_train)
result = sgd.score(x_test, y_test)

print(result)
