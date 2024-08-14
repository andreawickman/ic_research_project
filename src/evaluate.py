import numpy as np
import os 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error, mean_absolute_error


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

def evaluate_model(input_dir, feature_set_name, features, output_csv=None):
    # Number of folds
    n_folds = 5

    # Initialize lists to store all predictions and true values across all folds
    all_y_true = []
    all_y_pred = []

    # Initialize list to store fold-wise metrics
    fold_metrics = []

    # Loop through each fold
    for fold in range(1, n_folds + 1):
        # Load training and test set
        train = pd.read_csv(os.path.join(input_dir, f'training_data_fold{fold}.csv'))
        test = pd.read_csv(os.path.join(input_dir, f'test_data_fold{fold}.csv'))
        
        # Convert to log years
        y_train = np.log1p(train['TSI_days'] / 365)
        y_test = np.log1p(test['TSI_days'] / 365)

        # Specified feature set
        X_train = train[features]
        X_test = test[features]

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        
        # Predict on the test data
        y_pred = model.predict(X_test)
        
        # Store the true values and predictions
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        # Calculate and store metrics for the current fold
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Fold {fold}: MSE = {mse}, MAE = {mae}, R² = {r2}")
        
        fold_metrics.append({
            'Feature_Set': feature_set_name, 
            'Fold': fold,
            'MSE': mse,
            'MAE': mae,
            'R²': r2
        })

    # Convert lists to numpy arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # Calculate overall performance metrics across all folds
    overall_mse = mean_squared_error(all_y_true, all_y_pred)
    overall_mae = mean_absolute_error(all_y_true, all_y_pred)
    overall_r2 = r2_score(all_y_true, all_y_pred)

    # Print the overall performance metrics
    print("\nOverall Performance across all folds:")

    print(f"Overall MSE: {overall_mse}")
    print(f"Overall MAE: {overall_mae}")
    print(f"Overall R²: {overall_r2}")
    
    # Add overall metrics to the DataFrame
    fold_metrics.append({
        'Feature_Set': feature_set_name, 
        'Fold': 'Overall',
        'MSE': overall_mse,
        'MAE': overall_mae,
        'R²': overall_r2
    })

    metrics_df = pd.DataFrame(fold_metrics)
    # Save to CSV if output_csv is provided
    if output_csv:
        metrics_df.to_csv(output_csv, index=False)
        print(f"Performance metrics saved to {output_csv}")

    return metrics_df
