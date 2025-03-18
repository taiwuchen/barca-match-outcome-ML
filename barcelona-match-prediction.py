# FC Barcelona Match Outcome Prediction using Machine Learning
# ENSF 444 Project

"""
This notebook implements machine learning models to predict FC Barcelona match outcomes.
The models compare Logistic Regression, Random Forest, and SVM to predict whether Barcelona
will win, draw, or lose based on performance metrics.

Instructions:
1. Download the FC Barcelona Match Performance Dataset from Kaggle: 
   https://www.kaggle.com/datasets/adnanshaikh10/fc-barcelona-statistics
2. Save the CSV file in the same directory as this notebook
3. Run the notebook cells in order
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# 1. DATA LOADING AND EXPLORATION

# Load the dataset
def load_data(filepath='FC Barcelona Statistics.csv'):
    """Load the FC Barcelona match data from CSV."""
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        
        # Print columns to help debug
        print("\nDataset Columns:")
        print(df.columns.tolist())
        
        # Show a more detailed view of the first few rows
        print("\nDetailed view of first few rows:")
        pd.set_option('display.max_columns', None)
        print(df.head(2).to_string())
        pd.reset_option('display.max_columns')
        
        return df
    except FileNotFoundError:
        print(f"File not found at {filepath}. Please ensure the dataset is in the correct location.")
        return None
# Load the dataset
df = load_data()

# Display basic information about the dataset
print("\nDataset Information:")
print(df.info())

# Display the first few rows of the dataset
print("\nSample Data:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 2. EXPLORATORY DATA ANALYSIS

# Display distribution of match outcomes
plt.figure(figsize=(10, 6))
outcome_counts = df['Result'].value_counts()
sns.barplot(x=outcome_counts.index, y=outcome_counts.values)
plt.title('Distribution of Match Outcomes')
plt.xlabel('Match Result')
plt.ylabel('Count')
plt.show()

# Calculate win rate
win_rate = outcome_counts['Win'] / len(df) * 100
print(f"\nBarcelona's win rate in the dataset: {win_rate:.2f}%")

# Analyze key performance indicators
key_stats = ['PossessionPercentage', 'Shots', 'ShotsOnTarget', 'PassesCompleted', 
             'PassAccuracy', 'AerialDuelsWon', 'TacklesWon', 'Interceptions', 'Saves']

plt.figure(figsize=(15, 10))
for i, stat in enumerate(key_stats[:9], 1):
    if stat in df.columns:
        plt.subplot(3, 3, i)
        sns.boxplot(x='Result', y=stat, data=df)
        plt.title(f'{stat} by Match Result')

plt.tight_layout()
plt.show()

# Correlation analysis
if 'MatchResult_Encoded' not in df.columns:
    # Encode the target variable for correlation analysis
    result_map = {'Win': 2, 'Draw': 1, 'Loss': 0}
    df['MatchResult_Encoded'] = df['Result'].map(result_map)  # Change MatchResult to Result

# Select only numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = df[numeric_cols].corr()

# Plot correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 3. DATA PREPROCESSING

def preprocess_data(df):
    """Preprocess the data for machine learning."""
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Check if the dataframe is empty
    if data.empty:
        print("ERROR: Input dataframe is empty.")
        return None, None, None, None, []
    
    # First, print all available columns for debugging
    print("\nAll available columns in the dataset:")
    print(data.columns.tolist())
    
    # Check if 'Result' column exists or find an alternative
    result_column = None
    possible_result_columns = ['Result', 'MatchResult', 'Outcome', 'FTR', 'Match Result']
    
    for col in possible_result_columns:
        if col in data.columns:
            result_column = col
            print(f"Found result column: '{result_column}'")
            break
    
    if result_column is None:
        print("ERROR: Could not find a match result column.")
        print(f"Available columns: {data.columns.tolist()}")
        return None, None, None, None, []
    
    # Print unique values in Result column to verify
    print(f"\nUnique values in {result_column} column: {data[result_column].unique()}")
    
    # Define potential feature names (including common variations)
    potential_features = {
        'possession': ['PossessionPercentage', 'Possession', 'possession', 'PossessionPercent', 'Ball Possession %'],
        'shots': ['Shots', 'shots', 'TotalShots', 'Total Shots', 'Shot'],
        'shots_target': ['ShotsOnTarget', 'ShotOnTarget', 'Shots on Target', 'shots_on_target', 'shotsOnTarget'],
        'passes': ['PassesCompleted', 'Passes', 'CompletedPasses', 'PassesComplete', 'TotalPasses', 'Total Passes'],
        'pass_accuracy': ['PassAccuracy', 'PassAccuracyPercent', 'Pass Accuracy %', 'pass_accuracy'],
        'aerial_duels': ['AerialDuelsWon', 'AerialDuels', 'aerial_duels_won', 'Aerial Duels Won'],
        'tackles': ['TacklesWon', 'Tackles', 'tackles_won', 'TackleWon', 'Tackle'],
        'interceptions': ['Interceptions', 'Interception', 'interceptions'],
        'saves': ['Saves', 'GKSaves', 'GoalkeeperSaves', 'Save', 'Goalkeeper Saves'],
        'fouls': ['Fouls', 'FoulsCommitted', 'fouls', 'Foul'],
        'yellow_cards': ['YellowCards', 'Yellow Cards', 'yellowcards', 'Yellow', 'YellowCard'],
        'red_cards': ['RedCards', 'Red Cards', 'redcards', 'Red', 'RedCard'],
        'offsides': ['Offsides', 'Offside', 'offsides']
    }
    
    # Find matching columns in the dataset
    selected_features = []
    feature_mapping = {}  # Maps our standard names to the actual column names
    
    for feature_type, possible_names in potential_features.items():
        for name in possible_names:
            if name in data.columns:
                selected_features.append(name)
                feature_mapping[feature_type] = name
                break  # Once we find a match, move to the next feature type
    
    if not selected_features:
        print("ERROR: None of the specified features exist in the dataset.")
        print("Available columns are:", data.columns.tolist())
        return None, None, None, None, []
    
    print(f"Found {len(selected_features)} matching features:")
    for feature_type, column_name in feature_mapping.items():
        print(f"- Using '{column_name}' for {feature_type}")
    
    # Encode the target variable
    result_map = {'Win': 2, 'Draw': 1, 'Loss': 0}
    data['MatchResult_Encoded'] = data[result_column].map(result_map)
    
    # Verify encoding worked correctly
    if data['MatchResult_Encoded'].isna().sum() > 0:
        print("WARNING: Some Result values couldn't be mapped. Check your Result column values.")
        print(f"Unmapped values: {data[data['MatchResult_Encoded'].isna()][result_column].unique()}")
        
        # Fix encoding if needed (modify as appropriate for your dataset)
        for result in data[result_column].unique():
            if isinstance(result, str):  # Make sure it's a string before calling lower()
                if result.lower() == 'win' or 'win' in result.lower():
                    data.loc[data[result_column] == result, 'MatchResult_Encoded'] = 2
                elif result.lower() == 'draw' or result.lower() == 'tie' or 'draw' in result.lower():
                    data.loc[data[result_column] == result, 'MatchResult_Encoded'] = 1
                elif result.lower() == 'loss' or result.lower() == 'lose' or 'loss' in result.lower() or 'defeat' in result.lower():
                    data.loc[data[result_column] == result, 'MatchResult_Encoded'] = 0
    
    # Handle missing values - use mean for numeric columns
    for feature in selected_features:
        if data[feature].isnull().sum() > 0:
            print(f"Filling {data[feature].isnull().sum()} missing values in {feature}")
            data[feature].fillna(data[feature].mean(), inplace=True)
    
    # Create features and target arrays
    X = data[selected_features]
    y = data['MatchResult_Encoded']
    
    # Final check for NaN values
    if X.isna().sum().sum() > 0:
        print("WARNING: Features still contain NaN values after preprocessing. Filling remaining with 0.")
        X.fillna(0, inplace=True)
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, selected_features
    
# Preprocess the data
X_train, X_test, y_train, y_test, selected_features = preprocess_data(df)
print("\nChecking preprocessed data:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"Selected features: {selected_features}")

# Verify there are no NaN values
print(f"NaN values in X_train: {X_train.isna().sum().sum()}")
print(f"NaN values in X_test: {X_test.isna().sum().sum()}")
print(f"NaN values in y_train: {y_train.isna().sum()}")
print(f"NaN values in y_test: {y_test.isna().sum()}")
# 4. MODEL TRAINING AND EVALUATION

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple machine learning models."""
    # First, verify that we have valid data
    if X_train.shape[0] == 0 or y_train.shape[0] == 0:
        print("ERROR: Training data is empty. Cannot train models.")
        return {}
        
    if X_train.isna().sum().sum() > 0:
        print("WARNING: Training data contains NaN values. Filling with mean values.")
        X_train = X_train.fillna(X_train.mean())
        X_test = X_test.fillna(X_train.mean())  # Use training mean for test data too

    # Model definitions
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        
        'Random Forest': Pipeline([
            ('model', RandomForestClassifier(random_state=42))
        ]),
        
        'SVM (RBF Kernel)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(kernel='rbf', probability=True, random_state=42))
        ])
    }
    
    # Hyperparameter grids for tuning
    param_grids = {
        'Logistic Regression': {
            'model__C': [0.1, 1, 10],
            'model__solver': ['liblinear', 'saga']
        },
        
        'Random Forest': {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5]
        },
        
        'SVM (RBF Kernel)': {
            'model__C': [0.1, 1, 10, 100],
            'model__gamma': ['scale', 'auto', 0.1, 0.01]
        }
    }
    
    results = {}
    
    # Train and evaluate each model with cross-validation
    for name, pipeline in models.items():
        print(f"\nTraining {name}...")
        
        # Grid search for hyperparameter tuning
        grid_search = GridSearchCV(
            pipeline, 
            param_grids[name], 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        clf_report = classification_report(y_test, y_pred, 
                                          target_names=['Loss', 'Draw', 'Win'], 
                                          output_dict=True)
        
        # Cross-validation score
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Store results
        results[name] = {
            'model': best_model,
            'best_params': best_params,
            'accuracy': accuracy,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'classification_report': clf_report,
            'predictions': y_pred
        }
        
        print(f"{name} - Test Accuracy: {accuracy:.4f}")
        print(f"{name} - CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"Best parameters: {best_params}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=['Loss', 'Draw', 'Win'])}")
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Loss', 'Draw', 'Win'],
                   yticklabels=['Loss', 'Draw', 'Win'])
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    
    return results

# Train and evaluate models
model_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

# 5. MODEL COMPARISON AND VISUALIZATION

def compare_models(results):
    """Compare the performance of different models."""
    
    # Compare accuracies
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    cv_accuracies = [results[name]['cv_accuracy_mean'] for name in model_names]
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Test Accuracy')
    plt.bar(x + width/2, cv_accuracies, width, label='Cross-Validation Accuracy')
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(x, model_names)
    plt.legend()
    plt.ylim(0, 1)
    
    for i, v in enumerate(accuracies):
        plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    for i, v in enumerate(cv_accuracies):
        plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # Compare F1 scores for each class
    plt.figure(figsize=(15, 6))
    
    # Get F1 scores for each class
    class_names = ['Loss', 'Draw', 'Win']
    f1_scores = {
        model_name: [results[model_name]['classification_report'][class_name]['f1-score'] 
                    for class_name in class_names]
        for model_name in model_names
    }
    
    # Set width based on number of models and classes
    n_models = len(model_names)
    n_classes = len(class_names)
    width = 0.8 / n_models
    
    # Plot bars for each model and class
    for i, model_name in enumerate(model_names):
        offset = (i - n_models/2 + 0.5) * width
        plt.bar(np.arange(n_classes) + offset, f1_scores[model_name], width, label=model_name)
    
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Comparison by Class')
    plt.xticks(np.arange(n_classes), class_names)
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
    
    # Find the best model based on accuracy
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    print(f"\nBest performing model: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"Cross-Validation Accuracy: {results[best_model_name]['cv_accuracy_mean']:.4f} ± {results[best_model_name]['cv_accuracy_std']:.4f}")

    return best_model_name

# Compare model performances
best_model_name = compare_models(model_results)

# 6. FEATURE IMPORTANCE ANALYSIS

def analyze_feature_importance(model_results, best_model_name, features):
    """Analyze feature importance for the best model."""
    
    best_model = model_results[best_model_name]['model']
    
    # Extract the actual model from the pipeline if needed
    if hasattr(best_model, 'named_steps') and 'model' in best_model.named_steps:
        model = best_model.named_steps['model']
    else:
        model = best_model
    
    # Different methods to get feature importance based on model type
    if best_model_name == 'Logistic Regression':
        # For multi-class, we'll take the mean absolute coefficient across all classes
        if hasattr(model, 'coef_'):
            importance = np.mean(np.abs(model.coef_), axis=0)
        else:
            print("Could not extract coefficients from Logistic Regression model.")
            return
    
    elif best_model_name == 'Random Forest':
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            print("Could not extract feature importances from Random Forest model.")
            return
    
    elif best_model_name == 'SVM (RBF Kernel)':
        # SVMs don't have direct feature importance, so we'll use permutation importance
        # which is more computation-intensive but works for any model
        from sklearn.inspection import permutation_importance
        
        # This part might take some time to compute
        print("Computing permutation importance for SVM (this may take a few moments)...")
        perm_importance = permutation_importance(best_model, X_test, y_test, 
                                                n_repeats=10, random_state=42)
        importance = perm_importance.importances_mean
    else:
        print(f"Feature importance analysis not implemented for model type: {best_model_name}")
        return
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title(f'Feature Importance - {best_model_name}')
    plt.tight_layout()
    plt.show()
    
    print("\nFeature Importance:")
    for idx, row in feature_importance.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")
        
    return feature_importance

if __name__ == "__main__":
    try:
        # Load data
        df = load_data()
        
        if df is None or df.empty:
            print("ERROR: Failed to load valid data. Exiting.")
            exit(1)
        
        # Preprocess data
        X_train, X_test, y_train, y_test, selected_features = preprocess_data(df)
        
        if X_train is None:
            print("ERROR: Data preprocessing failed. Exiting.")
            exit(1)
            
        # Train models
        model_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        if not model_results:
            print("ERROR: Model training failed. Exiting.")
            exit(1)
            
        # Compare models
        best_model_name = compare_models(model_results)
        
        # Analyze feature importance for the best model
        analyze_feature_importance(model_results, best_model_name, selected_features)
        
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()