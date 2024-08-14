from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_selection import RFECV

def run_forest_class(X_train, y_train, X_test, y_test, predictors, n_estimators=100, max_depth = None, min_samples_leaf = 1, min_samples_split = 2, max_features = 'sqrt', random_state=42):
    # Select the predictor columns
    X_train_subset = X_train[predictors]
    X_test_subset = X_test[predictors]

    #DEFINE RANDOM FOREST CLASSIFIER INSTANCE   
    rf = RandomForestClassifier(n_estimators=n_estimators, 
                                random_state= random_state,
                                max_depth = max_depth,
                                max_features = max_features,
                                min_samples_leaf= min_samples_leaf,
                                min_samples_split= min_samples_split)
    
    #FIT RANDOM FOREST CLASSIFIER ON TRAINING DATA 
    rf.fit(X_train_subset, y_train)
    #MAKE PREDICTIONS ON TEST DATA
    y_pred = rf.predict(X_test_subset)

    #COMPUTE CLASSIFICATION ACCURACY 
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


## FEATURE SELECTION 
def feature_selection_and_plot(X_train, y_train, model_type, output_path):
    # Initialize and fit the Random Forest classifier
    rf = model_type(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Get feature importances from the fitted model
    features = X_train.columns
    feature_importances = list(zip(features, rf.feature_importances_))
    feature_importances.sort(key=lambda x: x[1])

    # Plot the feature importances
    plt.figure(figsize=(10, 8))
    plt.barh([x[0] for x in feature_importances], [x[1] for x in feature_importances])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance according to Random Forest')
    plt.savefig(output_path)
    plt.show()

    if model_type == RandomForestClassifier:
        cv = StratifiedKFold(n_splits=10)
        scoring = 'accuracy'
    elif model_type == RandomForestRegressor:
        cv = KFold(n_splits=10)
        scoring = 'r2'
    else:
        raise ValueError("Unsupported model type")

    # Use RFECV to perform feature selection with cross-validation
    rfe = RFECV(estimator=rf, step=1, cv=cv, scoring= scoring, n_jobs=-1)
    rfe.fit(X_train, y_train)

    # Get the selected features
    selected_features = X_train.columns[rfe.get_support()].tolist()

    print("Complete set of features:", list(features))
    print("Selected features:", selected_features)

    return selected_features