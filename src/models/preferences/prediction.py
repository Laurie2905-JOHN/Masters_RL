import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


def fit_random_forest_classifier(X, y):
    
    # Train the Random Forest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X, y)
    
    return rf_model

def predict_preference_using_model(child_features, ingredient_features, child, model, preprocessor):
    
    # Create a dictionary with combined features
    combined_features = [child_features | ingredient_features]
    
    # Create a DataFrame from the combined features dictionary
    df = pd.DataFrame(combined_features)
    
    X_preprocessed = preprocessor.transform(df)

    # Predict using the model
    y_pred = model.predict(X_preprocessed)
    
    return y_pred[0]

def ml_model_test(model, X_test, y_test, label_encoder):
    
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Print Metrics
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Precision: {:.2f}%".format(precision * 100))
    print("Recall: {:.2f}%".format(recall * 100))
    print("F1 Score: {:.2f}%".format(f1 * 100))
    print("Classification Report:\n", class_report)

    # Plot the confusion matrix using Seaborn
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()