#!/usr/bin/env python3
"""
FC Barcelona Match Outcome Prediction and Performance Suggestions

This script predicts the outcome (win/draw/loss) of FC Barcelona matches using historical
performance metrics (both numeric and categorical). In addition to training three machine
learning models (Logistic Regression, Random Forest, and Gradient Boosting), the script
extracts feature importances from non-linear models and provides suggestions for improving
the club’s performance based on the most influential KPIs.

Usage:
    Ensure that the dataset 'FC Barcelona Statistics.csv' is in the same directory as this script.
    Run the script using the command:
        python fc_barcelona_prediction_refined.py

Author: [Your Name]
Date: [Date]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_dataset(file_path):
    """
    Loads the dataset from a CSV file.
    
    Parameters:
        file_path (str): Path to the CSV dataset file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully. Here are the first few rows:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

def preprocess_data(df):
    """
    Preprocesses the dataset:
      - Separates the target variable ('Result').
      - For features, distinguishes between numeric and categorical columns.
      - Fills missing values (median for numeric, mode for categorical).
      - Encodes categorical features using one-hot encoding.
      - Scales numeric features.
      
    Parameters:
        df (pd.DataFrame): Original dataset.
    
    Returns:
        X (np.array): Preprocessed feature array.
        y (np.array): Encoded target labels.
        feature_names (list): List of feature names used after encoding.
    """
    # Check for target column
    if 'Result' not in df.columns:
        print("Error: 'Result' column not found in the dataset.")
        exit(1)
    
    # Separate target and features
    y = df['Result']
    X = df.drop(columns=['Result'])
    
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Fill missing values
    # For numeric columns, fill missing values with the median
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    # For categorical columns, fill missing values with the mode (most frequent value)
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])
    
    # Encode categorical features using one-hot encoding
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Record the final feature names
    feature_names = X_encoded.columns.tolist()
    
    # Encode target labels (win, draw, loss) into numeric values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print("Target classes:", label_encoder.classes_)
    
    # Scale numeric features only (one-hot encoded columns remain as is)
    # Identify the columns that were originally numeric after encoding
    scaler = StandardScaler()
    X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])
    
    return X_encoded.values, y_encoded, feature_names, label_encoder

def plot_confusion_matrix(cm, model_name, classes):
    """
    Plots a confusion matrix using matplotlib.
    
    Parameters:
        cm (np.array): Confusion matrix.
        model_name (str): Name of the model for title display.
        classes (list): List of class names.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Annotate cells with counts
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def train_and_evaluate(X_train, X_test, y_train, y_test, label_encoder, feature_names):
    """
    Trains and evaluates three models:
      - Logistic Regression
      - Random Forest Classifier
      - Gradient Boosting Classifier
      
    Evaluates each model using accuracy, weighted F1-score, and confusion matrix.
    Also extracts feature importance from non-linear models.
    
    Parameters:
        X_train, X_test (np.array): Feature sets.
        y_train, y_test (np.array): Target labels.
        label_encoder (LabelEncoder): Encoder to retrieve class names.
        feature_names (list): Names of the features.
    
    Returns:
        results (dict): Dictionary with evaluation metrics and feature importances.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"{name} Accuracy: {acc:.4f}")
        print(f"{name} Weighted F1-Score: {f1:.4f}")
        print(f"{name} Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
        results[name] = {
            'accuracy': acc,
            'f1_score': f1,
            'confusion_matrix': cm
        }
        
        # Plot confusion matrix for each model
        plot_confusion_matrix(cm, name, list(label_encoder.classes_))
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"{name} Cross-validation Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # If the model provides feature importances, record them
        if hasattr(model, 'feature_importances_'):
            results[name]['feature_importances'] = dict(zip(feature_names, model.feature_importances_))
    
    return results

def provide_suggestions(feature_importances, top_n=3):
    """
    Analyzes the top features and provides suggestions for improving club performance.
    
    Parameters:
        feature_importances (dict): Dictionary mapping feature names to importance scores.
        top_n (int): Number of top features to analyze.
    
    Returns:
        suggestions (list): List of performance improvement suggestions.
    """
    # Pre-defined mapping from feature keywords to actionable suggestions
    suggestion_mapping = {
        'possession': "Focus on strategies that improve ball retention and control the pace of the game.",
        'shots': "Improve shot quality and conversion rate to capitalize on offensive opportunities.",
        'pass': "Enhance passing accuracy to maintain possession and create better attacking plays.",
        'accuracy': "Work on overall accuracy in key areas such as passing or shooting.",
        'defense': "Strengthen defensive organization to minimize opponent scoring opportunities.",
        'turnover': "Reduce turnovers by improving decision-making in midfield.",
        'foul': "Address issues related to fouls to maintain discipline and avoid set-piece vulnerabilities."
    }
    
    # Sort features by importance
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    suggestions = []
    
    print("\nTop features and suggestions for performance improvement:")
    for feature, importance in sorted_features[:top_n]:
        # Lowercase the feature name for matching
        feature_lower = feature.lower()
        # Find a matching suggestion if a keyword is found in the feature name
        suggestion = None
        for key in suggestion_mapping:
            if key in feature_lower:
                suggestion = suggestion_mapping[key]
                break
        # Default suggestion if no keyword matches
        if suggestion is None:
            suggestion = "Investigate this factor further to determine how it can be optimized for better performance."
        
        suggestions.append((feature, importance, suggestion))
        print(f"- {feature}: importance = {importance:.4f}")
        print(f"  Suggestion: {suggestion}")
    
    return suggestions

def main():
    # Path to the dataset CSV file (ensure the file is in the same directory)
    dataset_path = 'FC Barcelona Statistics.csv'
    
    # Load the dataset
    df = load_dataset(dataset_path)
    
    # Preprocess data: include both numeric and categorical features, fill missing values, encode and scale
    X, y, feature_names, label_encoder = preprocess_data(df)
    
    # Split data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training and testing sets created.")
    
    # Train models and evaluate performance
    results = train_and_evaluate(X_train, X_test, y_train, y_test, label_encoder, feature_names)
    
    # For non-linear models, extract feature importances and provide suggestions
    # Here, we look at Random Forest and Gradient Boosting (if available)
    for model_name in ['Random Forest', 'Gradient Boosting']:
        if 'feature_importances' in results[model_name]:
            print(f"\nAnalyzing feature importances for {model_name}:")
            suggestions = provide_suggestions(results[model_name]['feature_importances'], top_n=3)
    
    # Summarize overall results
    print("\nFinal model comparison results:")
    for model_name, metrics in results.items():
        print(f"{model_name}: Accuracy = {metrics['accuracy']:.4f}, Weighted F1-score = {metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main()
