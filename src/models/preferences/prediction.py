import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Tuple
import copy

class PreferenceModel:
    def __init__(self, ingredient_df: pd.DataFrame, child_feature_data: Dict[str, Dict[str, Union[str, int]]], child_preference_data: Dict[str, Dict[str, Dict[str, List[str]]]], apply_SMOTE: bool = False, seed: Optional[int] = None):
        """
        Initialize the PreferenceModel class and prepare data for machine learning.
        """
        self.ingredient_df = ingredient_df
        self.child_feature_data = child_feature_data
        self.child_preference_data = copy.deepcopy(child_preference_data)
        self.apply_SMOTE = apply_SMOTE
        self.seed = seed

        self.ml_child_preferences = self.prepare_ml_preferences()
        self.X, self.y, self.known_df, self.preprocessor = self.prepare_ml_data()

        self.rf_model = RandomForestClassifier()
        self.rf_model.fit(self.X, self.y)

    def prepare_ml_preferences(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Prepare machine learning preferences from the child preference data and ingredient data. Removes the unknown from the preference list
        and adds it to an unknown list.
        """
        ml_preferences = {}

        for child in self.child_preference_data:
            ml_preferences[child] = {}
            ml_preferences[child]['likes'] = self.child_preference_data[child]['known']['likes']
            ml_preferences[child]['neutral'] = self.child_preference_data[child]['known']['neutral']
            ml_preferences[child]['dislikes'] = self.child_preference_data[child]['known']['dislikes']

            # Determine unknown features
            known_ingredients = set(self.child_preference_data[child]['known']['likes'] + self.child_preference_data[child]['known']['neutral'] + self.child_preference_data[child]['known']['dislikes'])
            all_ingredients = set(self.ingredient_df['Category7'])
            unknown_ingredients = all_ingredients - known_ingredients

            ml_preferences[child]['unknown'] = list(unknown_ingredients)

        return ml_preferences

    def prepare_ml_data(self) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, ColumnTransformer]:
        """
        Prepare the data for machine learning.
        """
        child_df = pd.DataFrame.from_dict(self.child_feature_data, orient='index').reset_index().rename(columns={'index': 'child'})
        preferences_df = pd.DataFrame(self.ml_child_preferences).T.reset_index().rename(columns={'index': 'child'})

        likes_df = self.explode_preferences(preferences_df, 'likes', 2)
        neutral_df = self.explode_preferences(preferences_df, 'neutral', 1)
        dislikes_df = self.explode_preferences(preferences_df, 'dislikes', 0)
        unknown_df = self.explode_preferences(preferences_df, 'unknown', np.nan)

        preferences_long_df = pd.concat([likes_df, neutral_df, dislikes_df, unknown_df])

        combined_df = pd.merge(child_df, preferences_long_df, on='child')
        df = pd.merge(combined_df, self.ingredient_df[['Category7', 'Category1', 'Colour', 'Texture', 'Taste', 'Healthy']], left_on='ingredient', right_on='Category7')

        df.drop(columns=['Category7'], inplace=True)
        df.rename(columns={
            'ingredient': 'ingredient',
            'Category1': 'type',
            'Colour': 'colour',
            'Texture': 'texture',
            'Taste': 'taste',
            'Healthy': 'healthy'
        }, inplace=True)

        df = df[['age', 'gender', 'health_consideration', 'favorite_cuisine', 'ingredient', 'type', 'colour', 'texture', 'taste', 'healthy', 'preference']]

        # Dropping unknown nan preferences
        known_df = df.dropna(subset=['preference'])

        preprocessor = self.get_data_preprocessor()
        # Fit the preprocessor on all data so the processor sees all categories
        preprocessor.fit(df.drop(columns=['preference']))

        X = known_df.drop(columns=['preference'])
        y = known_df["preference"].values

        preprocessor.fit(X)

        # Apply the transformations and prepare the dataset
        X_transformed = preprocessor.transform(X)
    
        if self.apply_SMOTE:
            X, y = self.apply_smote(X_transformed, y)
        else:
            X = X_transformed

        return X, y, known_df, preprocessor

    @staticmethod
    def explode_preferences(preferences_df: pd.DataFrame, column: str, label: Union[int, float]) -> pd.DataFrame:
        """
        Explode a preferences column into a long format DataFrame with a specific label.
        """
        exploded_df = preferences_df.explode(column)[['child', column]].rename(columns={column: 'ingredient'})
        exploded_df['preference'] = label
        return exploded_df

    @staticmethod
    def get_data_preprocessor() -> ColumnTransformer:
        """
        Get the preprocessor for the data.
        """
        preprocessor = ColumnTransformer(
            transformers=[
                ("age", OneHotEncoder(), ["age"]),
                ("gender", OneHotEncoder(), ["gender"]),
                ("health_consideration", OneHotEncoder(), ["health_consideration"]),
                ("favorite_cuisine", OneHotEncoder(), ["favorite_cuisine"]),
                ("type", OneHotEncoder(), ["type"]),
                ("colour", OneHotEncoder(), ["colour"]),
                ("taste", OneHotEncoder(), ["taste"]),
                ("texture", OneHotEncoder(), ["texture"]),
                ("healthy", OneHotEncoder(), ["healthy"]),
            ],
        )
        return preprocessor

    @staticmethod
    def apply_smote(X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Apply SMOTE to balance the classes.
        """
        # Convert sparse matrix to dense format
        X_dense = X.toarray() if hasattr(X, 'toarray') else X
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_dense, y)
        return X_res, y_res

    def combine_features(self, ingredient: str, child: str) -> pd.DataFrame:
        """
        Combine features from ingredient and child data into a single DataFrame.
        """
        ingredient_details = self.ingredient_df[self.ingredient_df['Category7'] == ingredient].iloc[0]

        ingredient_features = {
            "ingredient": ingredient,
            "type": ingredient_details["Category1"],
            "texture": ingredient_details["Texture"],
            "colour": ingredient_details["Colour"],
            "taste": ingredient_details["Taste"],
            "healthy": ingredient_details["Healthy"]
        }

        child_features = {
            "age": self.child_feature_data[child]["age"],
            "gender": self.child_feature_data[child]["gender"],
            "health_consideration": self.child_feature_data[child]["health_consideration"],
            "favorite_cuisine": self.child_feature_data[child]["favorite_cuisine"]
        }

        combined_features = {**child_features, **ingredient_features}
        df = pd.DataFrame([combined_features])
        return df

    def predict_preference_using_model(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict preferences using the trained model and preprocessor.
        """
        X_preprocessed = self.preprocessor.transform(df)
        y_pred = self.rf_model.predict(X_preprocessed)
        return y_pred

    @staticmethod
    def get_true_preference_label(child_preference_data: Dict[str, Dict[str, Dict[str, List[str]]]], ingredient: str, child: str) -> int:
        """
        Get the true preference label for a given ingredient.
        """
        if ingredient in child_preference_data[child]['unknown']["likes"] + child_preference_data[child]['known']["likes"]:
            return 2
        elif ingredient in child_preference_data[child]['unknown']["neutral"] + child_preference_data[child]['known']["neutral"]:
            return 1
        elif ingredient in child_preference_data[child]['unknown']["dislikes"] + child_preference_data[child]['known']["dislikes"]:
            return 0
        else:
            raise ValueError(f"Ingredient {ingredient} not found in preferences")

    def ml_model_test(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Test the trained model and print evaluation metrics and confusion matrix.
        """
        y_pred = self.rf_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        print("Accuracy: {:.2f}%".format(accuracy * 100))
        print("Precision: {:.2f}%".format(precision * 100))
        print("Recall: {:.2f}%".format(recall * 100))
        print("F1 Score: {:.2f}%".format(f1 * 100))
        print("Classification Report:\n", class_report)

        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    def run_pipeline(self) -> None:
        """
        Run the entire preference prediction and evaluation pipeline.
        """
        total_predicted_preferences = []
        total_true_preferences = []
        child_predictions = {}
        child_true_labels = {}
        batch_data = []
        batch_labels = []
        child_indices = []

        for child, preferences in self.ml_child_preferences.items():
            child_predictions[child] = []
            child_true_labels[child] = []
            for ingredient in preferences['unknown']:
                child_and_ingredient_feature_df = self.combine_features(ingredient, child)
                batch_data.append(child_and_ingredient_feature_df)
                true_label = self.get_true_preference_label(self.child_preference_data, ingredient, child)
                batch_labels.append(true_label)
                child_indices.append(child)

        if batch_data:
            batch_df = pd.concat(batch_data, ignore_index=True)
            batch_predictions = self.predict_preference_using_model(batch_df)

            if not isinstance(batch_predictions, (list, pd.Series, np.ndarray)):
                batch_predictions = [batch_predictions]

            for i, predicted_preference in enumerate(batch_predictions):
                child = child_indices[i]
                child_predictions[child].append(predicted_preference)
                child_true_labels[child].append(batch_labels[i])
                total_predicted_preferences.append(predicted_preference)
                total_true_preferences.append(batch_labels[i])

        # for child in child_predictions:
            # if child_predictions[child]:
                # child_class_report = classification_report(child_true_labels[child], child_predictions[child])
                # logging.info(f"{child} Classification Report:\n{child_class_report}")

        total_class_report = classification_report(total_true_preferences, total_predicted_preferences)
        logging.info(f"Total Classification Report:\n{total_class_report}")

        updated_preferences = self.update_preferences_with_predictions(child_predictions)
        
        return updated_preferences

    def update_preferences_with_predictions(self, child_predictions: Dict[str, List[int]]) -> Dict[str, Dict[str, List[str]]]:
        """
        Update preferences with the model predictions.
        """
        updated_preferences = {}
        for child, preferences in self.ml_child_preferences.items():
            predicted_likes = {
                ingredient: 'like' if pred == 2 else ('neutral' if pred == 1 else 'dislike')
                for ingredient, pred in zip(preferences['unknown'], child_predictions[child])
            }

            if child not in updated_preferences:
                updated_preferences[child] = {
                    'likes': preferences.get('likes', []),
                    'neutral': preferences.get('neutral', []),
                    'dislikes': preferences.get('dislikes', [])
                }

            for ingredient, pred in zip(preferences['unknown'], child_predictions[child]):
                if pred == 2:
                    updated_preferences[child]['likes'].append(ingredient)
                elif pred == 1:
                    updated_preferences[child]['neutral'].append(ingredient)
                else:
                    updated_preferences[child]['dislikes'].append(ingredient)

        return updated_preferences

