# tune.py
from sklearn.model_selection import GridSearchCV
from models import get_model

def tune_model(X_train, y_train):
    param_grid = {
        "C": [0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"]
    }
    grid = GridSearchCV(get_model(), param_grid, cv=3)
    grid.fit(X_train, y_train)
    print("Best Params:", grid.best_params_)
    return grid.best_estimator_
