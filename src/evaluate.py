import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, accuracy_score


def calculate_class_accuracies(train_features, predictors):
    class_accuracies = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_index, test_index in skf.split(train_features, train_features['TSI_encoded']):
        X_train, X_test = train_features.iloc[train_index][predictors], train_features.iloc[test_index][predictors]
        y_train, y_test = train_features.iloc[train_index]['TSI_encoded'], train_features.iloc[test_index]['TSI_encoded']
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        for label in np.unique(y_test):
            if label not in class_accuracies:
                class_accuracies[label] = []
                
            class_indices = y_test == label
            accuracy = accuracy_score(y_test[class_indices], y_pred[class_indices])
            class_accuracies[label].append(accuracy)
    
    class_accuracies_mean = {label: np.mean(acc) for label, acc in class_accuracies.items()}
    class_accuracies_ci = {label: np.percentile(acc, [2.5, 97.5]) for label, acc in class_accuracies.items()}
    
    return class_accuracies_mean, class_accuracies_ci

# Function to calculate R2 scores and confidence intervals
def calculate_r2_scores(train_features, predictors):
    r2_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_index, test_index in skf.split(train_features, train_features['TSI_encoded']):
        X_train, X_test = train_features.iloc[train_index][predictors], train_features.iloc[test_index][predictors]
        y_train, y_test = train_features.iloc[train_index]['TSI_days'], train_features.iloc[test_index]['TSI_days']
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)
    
    mean_r2 = np.mean(r2_scores)
    r2_ci = np.percentile(r2_scores, [2.5, 97.5])
    
    return mean_r2, r2_ci