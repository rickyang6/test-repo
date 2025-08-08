
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


class IrisClassifier:
    """A simple classifier for the Iris dataset using logistic regression.
    
        This class provides methods to:
        - Load and preprocess the Iris dataset
        - Engineer basic features
        - Train a logistic regression model
        - Evaluate performance using AUC scores
        - Retrieve feature importances
    """
    def __init__(self, target_column="species", model=LogisticRegression(), test_size=0.2):
        self.target_column = target_column
        self.model = model
        self.test_size = test_size
        
        self.data = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.feature_importance = None

    def load_data(self, target_column=None, binary_target=None):
        """Load iris dataset and store in self.data."""
        if target_column is None:
            target_column = self.target_column

        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

        if binary_target:
            df[target_column] = (iris.target == 0).astype(int)
        else: 
            df[target_column] = iris.target_names[iris.target]
        
        self.data = df
        return df
 
    def prepare_data(self, target_column=None, test_size=None):
        """Split data into training and test sets."""
        if target_column is None:
            target_column = self.target_column
        if test_size is None:
            test_size = self.test_size

        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size
        )

    def train(self, model=None):
        """Train the model."""
        if model is None:
            model = self.model
        else:
            self.model = model

        model.fit(self.X_train, self.y_train)

    def evaluate(self):
        """Evaluate the model and return accuracy."""
        y_pred = self.model.predict(self.X_test)
        return roc_auc_score(self.y_test, y_pred)

    def get_feature_importance(self):
        """Get feature importance (coefficients for logistic regression)."""
        if hasattr(self.model, "coef_"):
            self.feature_importance = pd.DataFrame({
                "feature": self.X_train.columns,
                "importance": self.model.coef_[0]
            }).sort_values(by="importance", ascending=False)
            return self.feature_importance
        else:
            raise ValueError("The current model does not support feature importance.")


if __name__ == "__main__":
    clf = IrisClassifier()
    clf.load_data(binary_target="setosa")
    clf.prepare_data()
    clf.train()
    acc = clf.evaluate()
    print(f"AUC: {acc:.2f}")
    print(clf.get_feature_importance())
