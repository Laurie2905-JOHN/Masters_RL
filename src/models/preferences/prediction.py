import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import copy
import prince
import os 
import logging 
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from matplotlib.patches import Ellipse

class PreferenceModel:
    def __init__(self, ingredient_df: pd.DataFrame, child_feature_data: Dict[str, Dict[str, Union[str, int]]], child_preference_data: Dict[str, Dict[str, Dict[str, List[str]]]], apply_SMOTE: bool = False, visualize_data: bool = False, file_path: str = None, seed: Optional[int] = None):
        """
        Initialize the PreferenceModel class and prepare data for machine learning.
        """
        self.ingredient_df = ingredient_df
        self.child_feature_data = child_feature_data
        self.child_preference_data = copy.deepcopy(child_preference_data)
        self.apply_SMOTE = apply_SMOTE
        self.seed = seed

        self.ml_child_preferences = self.prepare_ml_preferences()
        self.X, self.y, self.preprocessor = self.prepare_ml_data()
        
        if visualize_data:
            self.visualize_complete_data(file_path)

        self.rf_model = RandomForestClassifier()
        self.rf_model.fit(self.X, self.y)

    @staticmethod
    def plot_ellipse(ax, data, color, label):
        """
        Plot an ellipse around the data points corresponding to a particular cluster or class label.
        """
        if len(data) > 1:
            cov = np.cov(data, rowvar=False)
            mean = np.mean(data, axis=0)

            # Eigen decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

            # Calculate the angle of the ellipse
            vx, vy = eigenvectors[:, 0]
            theta = np.arctan2(vy, vx)

            # Width and height of the ellipse
            width, height = 2 * np.sqrt(eigenvalues)
            
            ellipse = Ellipse(mean, width, height, angle=np.degrees(theta), edgecolor=color, facecolor='none', label=label)
            ax.add_patch(ellipse)

    def visualize_complete_data(self, file_path) -> None:
        """
        Visualize the complete data using MCA and save the plot to a file.
        """
        df = self.get_complete_preference_df()
        
        # Separate the features and class labels
        features = df.drop(columns=['preference', 'known_unknown', 'child'])
        class_labels = df['preference']

        # Apply MCA
        mca = prince.MCA(n_components=2, n_iter=30, copy=True, check_input=True, engine='sklearn', random_state=42)
        mca = mca.fit(features)

        # Transform the data
        df_mca = mca.transform(features)
        df_mca['class_label'] = class_labels.values

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(df_mca[[0, 1]])
        df_mca['cluster'] = clusters

        # Calculate clustering accuracy
        clustering_accuracy = adjusted_rand_score(df_mca['class_label'], df_mca['cluster'])

        # Define color palette
        palette = ['red', 'black', 'blue']
        cluster_colors = [palette[label] for label in df_mca['cluster']]
        
        # Define a distinct color palette for class labels
        unique_labels = class_labels.unique()
        label_palette = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9', '#999999']
        color_dict = {label: label_palette[i % len(label_palette)] for i, label in enumerate(unique_labels)}
        class_colors = [color_dict[label] for label in df_mca['class_label']]

        # Create subplots
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

        # Plot MCA results with class labels
        axes[0].scatter(df_mca[0], df_mca[1], c=class_colors, alpha=0.6)
        handles_class = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[label], markersize=10, label=label) for label in unique_labels]
        axes[0].legend(title='Class Label', handles=handles_class)
        axes[0].set_title('MCA of One-Hot Encoded Data with Class Labels')
        axes[0].set_xlabel('MCA 1')
        axes[0].set_ylabel('MCA 2')

        # Draw ellipses for class labels
        for label in unique_labels:
            data_points = df_mca[df_mca['class_label'] == label][[0, 1]].values
            self.plot_ellipse(axes[0], data_points, color_dict[label], label)

        # Plot MCA results with cluster labels
        axes[1].scatter(df_mca[0], df_mca[1], c=cluster_colors, alpha=0.6)
        handles_cluster = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=10, label=f'Cluster {i}') for i in range(3)]
        axes[1].legend(title='Cluster', handles=handles_cluster)
        axes[1].set_title('MCA of One-Hot Encoded Data with KMeans Clusters')
        axes[1].set_xlabel('MCA 1')
        axes[1].set_ylabel('MCA 2')

        # Draw ellipses for clusters
        for cluster in range(3):
            data_points = df_mca[df_mca['cluster'] == cluster][[0, 1]].values
            self.plot_ellipse(axes[1], data_points, palette[cluster], f'Cluster {cluster}')

        # Calculate and display variance explained by each component
        eigenvalues = mca.eigenvalues_
        variance_explained = [val / sum(eigenvalues) for val in eigenvalues]
        fig.suptitle(f'MCA of One-Hot Encoded Data\nVariance Explained: MCA 1 = {variance_explained[0]:.2%}, MCA 2 = {variance_explained[1]:.2%}\nClustering Accuracy: {clustering_accuracy:.2%}', fontsize=16)

        if file_path:
            # Save the plot
            plt.savefig(file_path)
            plt.close()
        else:
            plt.show()
        
    def get_complete_preference_df(self) -> pd.DataFrame:
        """
        Get the complete ingredient DataFrame with all features, including known/unknown status.
        """
        data = []

        for child, prefs in self.child_preference_data.items():
            for category in ['likes', 'neutral', 'dislikes']:
                known_prefs = prefs.get('known', {}).get(category, [])
                unknown_prefs = prefs.get('unknown', {}).get(category, [])
                
                for ingredient in known_prefs + unknown_prefs:
                    data.append({
                        'child': child,
                        'ingredient': ingredient,
                        'preference': category,
                        'known_unknown': 'known' if ingredient in known_prefs else 'unknown'
                    })
            
        # Create DataFrame
        preferences_df = pd.DataFrame(data)

        # Add child features
        child_df = pd.DataFrame.from_dict(self.child_feature_data, orient='index').reset_index().rename(columns={'index': 'child'})
        combined_df = pd.merge(preferences_df, child_df, on='child', how='left')

        # Add ingredient features
        combined_df = pd.merge(combined_df, self.ingredient_df[['Category7', 'Category1', 'Colour', 'Texture', 'Taste', 'Healthy']],
                            left_on='ingredient', right_on='Category7', how='left')

        combined_df.drop(columns=['Category7'], inplace=True)
        combined_df.rename(columns={
            'Category1': 'type',
            'Colour': 'colour',
            'Texture': 'texture',
            'Taste': 'taste',
            'Healthy': 'healthy'
        }, inplace=True)

        combined_df = combined_df[['child', 'age', 'gender', 'health_consideration', 'favorite_cuisine', 'ingredient', 'type', 'colour', 'texture', 'taste', 'healthy', 'preference', 'known_unknown']]
        
        return combined_df

    def prepare_ml_preferences(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Prepare machine learning preferences from the child preference data and ingredient data.
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
        # Use the combined DataFrame from the refactored method
        df = self.get_complete_preference_df()

        # Dropping unknown preferences
        known_df = df[df['known_unknown'] == 'known'].drop(columns=['known_unknown', 'child'])

        preprocessor = self.get_data_preprocessor()
        
        # Fit the preprocessor on all data so the processor sees all categories except the class label and preference
        preprocessor.fit(df.drop(columns=['preference', 'known_unknown']))

        X = known_df.drop(columns=['preference'])
        y = known_df["preference"].values

        preprocessor.fit(X)

        # Apply the transformations and prepare the dataset
        X_transformed = preprocessor.transform(X)

        if self.apply_SMOTE:
            X, y = self.apply_smote(X_transformed, y)
        else:
            X = X_transformed

        return X, y, preprocessor

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
    def get_true_preference_label(child_preference_data: Dict[str, Dict[str, Dict[str, List[str]]]], ingredient: str, child: str) -> str:
        """
        Get the true preference label for a given ingredient.
        """
        if ingredient in child_preference_data[child]['unknown']["likes"] + child_preference_data[child]['known']["likes"]:
            return 'likes'
        elif ingredient in child_preference_data[child]['unknown']["neutral"] + child_preference_data[child]['known']["neutral"]:
            return 'neutral'
        elif ingredient in child_preference_data[child]['unknown']["dislikes"] + child_preference_data[child]['known']["dislikes"]:
            return 'dislikes'
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
                updated_preferences[child][pred].append(ingredient)

        return updated_preferences
