from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier


class ML:
    def __init__(self, learn_df):
        self.learn_df = learn_df
        self.models = []

    def learn_model(self):
        svm = SVC(probability=True, kernel='linear')
        lr = LogisticRegression(**{'C': 0.1,
                                   'max_iter': 2000,
                                   'penalty': 'l1',
                                   'solver': 'liblinear'})
        voting_clf = VotingClassifier(estimators=[('lr', lr), ('cv', svm)], voting='soft')
        scale = StandardScaler()
        X_train = self.learn_df[['AVG_1', 'AVG_2', 'diff', 'VOL_1', 'VOL_2']]
        X_train = scale.fit_transform(X_train)
        y_train = self.learn_df['indicator']
        svm.fit(X_train, y_train)
        lr.fit(X_train, y_train)
        voting_clf.fit(X_train, y_train)
        self.models.append(svm), self.models.append(lr), self.models.append(voting_clf)

    def predict(self, min_df):
        X_final = min_df[['AVG_1', 'AVG_2', 'diff', 'VOL_1', 'VOL_2']]
        scale = StandardScaler()
        X_final = scale.fit_transform(X_final)
        return {
            "SVM": self.models[0].predict(X_final),
            "Logistic Regression": self.models[1].predict(X_final),
            "Ensemble": self.models[2].predict(X_final)
        }
