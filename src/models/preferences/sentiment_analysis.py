from transformers import pipeline
import torch
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from models.preferences.prediction import PreferenceModel
import random
from typing import Dict, List, Tuple, Union, Optional
import copy
import numpy as np

class SentimentAnalyzer:
    def __init__(self, true_preferences, menu_plan, model_name: str = "finiteautomata/bertweet-base-sentiment-analysis", seed: int = 42):
        """
        Initialize the sentiment analyzer with the specified model.
        """
        self.device = 'cpu'
        self.true_preferences = copy.deepcopy(true_preferences)
        self.sentiment_analyzer = pipeline('sentiment-analysis', model=model_name, device=self.device)
        self.menu_plan = [ingredient for ingredient in menu_plan.values()]
        self.label_mapping = {'likes': 2, 'neutral': 1, 'dislikes': 0}
        self.feedback = self.get_feedback()
        self.changes = []
        self.incorrect_comments = []
        
    def analyze_sentiment(self, comment: str) -> str:
        """
        Analyze the sentiment of a given comment and return the corresponding label.
        """
        result = self.sentiment_analyzer(comment)
        label = result[0]['label']

        if label == 'POS':
            return 'likes'
        elif label == 'NEG':
            return 'dislikes'
        elif label == 'NEU':
            return 'neutral'
        else:
            raise ValueError(f"Unknown sentiment label: {label}")

    def get_sentiment_and_update_data(self, plot_confusion_matrix: bool = False) -> Tuple[List[Dict[str, Union[str, List[str]]]], Dict[str, Dict[str, List[str]]], float, List[Dict[str, str]]]:
        # Preference input here are the predicted ones and are updated with the feedback
        """
        Analyze feedback, update preferences, and return changes, updated preferences, accuracy, and incorrect comments.
        """
        true_labels = []
        pred_labels = []
        feedback = self.feedback

        for child, fb in feedback.items():
            comments = re.split(r'[,.!?]', fb["comment"].lower())
            correct_action = fb["correct_action"]

            for sentence in comments:
                if sentence.strip():
                    pred_label = self.analyze_sentiment(sentence.strip())

                    for ingredient in self.menu_plan:
                        if ingredient.lower() in sentence:
                            current_preference = self.get_current_preference_label(child, ingredient)
                            
                            if current_preference != pred_label:
                                self.changes.append(self.update_preferences(child, ingredient, current_preference, pred_label))

                            true_labels.append(correct_action[ingredient])
                            pred_labels.append(pred_label)

                            if pred_label != correct_action[ingredient]:
                                self.incorrect_comments.append({
                                    "child": child,
                                    "comment": sentence,
                                    "predicted": pred_label,
                                    "actual": correct_action[ingredient]
                                })

        accuracy = self.calculate_accuracy(true_labels, pred_labels)

        if plot_confusion_matrix and true_labels:
            self.plot_confusion_matrix(true_labels, pred_labels)

        return self.true_preferences, accuracy, feedback

    def get_current_preference_label(self, child: str, ingredient: str) -> Optional[str]:
        """
        Get the current preference list (likes, dislikes, neutral) for a given ingredient and updated the known list with feedback.
        """ 
        if ingredient in self.true_preferences[child]['known']["likes"]:
            return "likes"
        elif ingredient in self.true_preferences[child]['known']["dislikes"]:
            return "dislikes"
        elif ingredient in self.true_preferences[child]['known']["neutral"]:
            return "neutral"
        elif ingredient in self.true_preferences[child]['unknown']["likes"] + self.true_preferences[child]['unknown']["dislikes"]  + self.true_preferences[child]['unknown']["neutral"]: 
            return "unknown"
        else: 
            raise ValueError(f"Ingredient {ingredient} not found in preferences for child {child}")
        
    def update_preferences(self, child: str, ingredient: str, current_preference: Optional[str], new_preference: str) -> Dict[str, Union[str, List[str]]]:
        """
        Update the preferences of a child for a given ingredient and return the change details.
        In Python, dictionaries are mutable objects, so when you modify them within a function, the changes are reflected outside the function as well.
        Therefore no need to return preferences
        """
        change = {"child": child, "ingredient": ingredient, "change": ""}
        if current_preference:
            if current_preference == "unknown":
                for category in self.true_preferences[child]['unknown']:
                    if ingredient in self.true_preferences[child]['unknown'][category]:
                        self.true_preferences[child]['unknown'][category].remove(ingredient)
                        self.true_preferences[child]['known'][new_preference].append(ingredient)
                        break
            else:
                self.true_preferences[child]['known'][current_preference].remove(ingredient)
                self.true_preferences[child]['known'][new_preference].append(ingredient)
                
            change["change"] += f"removed from {current_preference}"

        
        if change["change"]:
            change["change"] += ", "
        change["change"] += f"added to {new_preference}"
        return change

    def calculate_accuracy(self, true_labels: List[str], pred_labels: List[str]) -> float:
        """
        Calculate the accuracy of the sentiment predictions.
        """
        correct_actions = sum(1 for true, pred in zip(true_labels, pred_labels) if true == pred)
        total_actions = len(true_labels)
        return correct_actions / total_actions if total_actions > 0 else 0

    def plot_confusion_matrix(self, true_labels: List[str], pred_labels: List[str]) -> None:
        """
        Plot the confusion matrix for the sentiment predictions.
        """
        cm = confusion_matrix([self.label_mapping[label] for label in true_labels],
                              [self.label_mapping[label] for label in pred_labels],
                              labels=[0, 1, 2])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['likes', 'neutral', 'dislikes'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.show()

    @staticmethod
    def generate_random_indices(n, mean_length, std_dev_length):
        
        # Generate a random length for the list
        length = int(np.random.normal(loc=mean_length, scale=std_dev_length))
        
        # Clamp the length to ensure it is within a reasonable range [0, n]
        length = max(0, min(n, length))
        
        # Generate a list of unique random indices of the specified length
        indices = np.random.choice(n, size=length, replace=False)
        
        return indices.tolist()

    def get_feedback(self, average_exclude: int = 5, std_dev_exclude: int = 2, seed: Optional[int] = None) -> Dict[str, Dict[str, Union[str, Dict[str, str]]]]:

        """
        Generate feedback based on true_preferences (preferences are the true initialized ones, no error) and menu plan,
        providing randomized comments on the ingredients. Exclude an average of average_exclude children with a standard deviation of std_dev_exclude.
        """
        
        comments = [
            # List of comment templates and their corresponding feedback types
            ("Didn't like the {} and {} in the dish, but the {} was tasty.", ["dislikes", "dislikes", "likes"]),
            ("Did not enjoy the {} and {}.", ["dislikes", "dislikes"]),
            ("Enjoyed the {} and {}, but was okay with the {}.", ["likes", "likes", "neutral"]),
            ("Loved the {}, but didn't like the {} and {}.", ["likes", "dislikes", "dislikes"]),
            ("The {} was great, but the {} was just okay.", ["likes", "neutral"]),
            ("Didn't enjoy the {}, but the {} was okay.", ["dislikes", "neutral"]),
            ("Loved the {} and {}, but not the {}.", ["likes", "likes", "dislikes"]),
            ("Loved the {}, but the {} was not appealing.", ["likes", "dislikes"]),
            ("Enjoyed the {}, but the {} was not liked.", ["likes", "dislikes"]),
            ("Didn't like the {}, {} and {} together.", ["dislikes", "dislikes", "dislikes"]),
            ("Really liked the {} with {} and the {} was tasty.", ["likes", "likes", "likes"]),
            ("Didn't like the {} in the dish, but the {} was fine.", ["dislikes", "neutral"]),
            ("Enjoyed the {} and {}, but not the {}.", ["likes", "likes", "dislikes"]),
            ("Didn't like the {} and {}.", ["dislikes", "dislikes"]),
            ("The {} and {} were amazing, but didn't enjoy the {} much.", ["likes", "likes", "dislikes"]),
            ("Loved the {} and {}, but not the {}.", ["likes", "likes", "dislikes"]),
            ("Didn't enjoy the {} much, but the {} was okay.", ["dislikes", "neutral"]),
            ("The {} and {} dish was great.", ["likes", "likes"]),
            ("Didn't like the {}.", ["dislikes"]),
            ("Enjoyed the {} and {}.", ["likes", "likes"]),
            ("Loved the {} and {}.", ["likes", "likes"]),
            ("Didn't like the {} and the {}.", ["dislikes", "dislikes"]),
            ("Enjoyed the {} and {}, but the {} was okay.", ["likes", "likes", "neutral"]),
            ("Didn't like the {} and {} in the dish.", ["dislikes", "dislikes"]),
            ("Didn't like the {}, but the {} was okay.", ["dislikes", "neutral"]),
            ("Enjoyed the {} and {}, but didn't like the {}.", ["likes", "likes", "dislikes"]),
            ("Didn't like the {}.", ["dislikes"]),
            ("Loved the {} and {}, but the {} was not liked.", ["likes", "likes", "dislikes"]),
            ("Didn't like the {}, but the {} was okay.", ["dislikes", "neutral"]),
            ("Enjoyed the {} and {}.", ["likes", "likes"]),
            ("Liked the {} but not the {}.", ["likes", "dislikes"]),
            ("The {} was fine, but the {} wasn't good.", ["neutral", "dislikes"]),
            ("The {} and {} were great, but the {} was not.", ["likes", "likes", "dislikes"]),
            ("The {} was tasty, but the {} wasn't.", ["likes", "dislikes"]),
            ("The {} was okay, but the {} wasn't appealing.", ["neutral", "dislikes"]),
            ("Didn't like the {}, but the {} was good.", ["dislikes", "likes"]),
            ("The {} and {} were okay, but the {} wasn't.", ["neutral", "neutral", "dislikes"]),
            ("Really liked the {}, but the {} was too strong.", ["likes", "dislikes"]),
            ("Enjoyed the {}, but the {} was too bland.", ["likes", "dislikes"]),
            ("The {} was fine, but the {} needed more flavor.", ["neutral", "dislikes"]),
            ("Loved the {}, but the {} was not good.", ["likes", "dislikes"]),
            ("Didn't enjoy the {}, but the {} was okay.", ["dislikes", "neutral"]),
            ("The {} was good, but the {} was not to my taste.", ["likes", "dislikes"]),
            ("Enjoyed the {}, but the {} was too overpowering.", ["likes", "dislikes"]),
            ("The {} was delicious, but the {} wasn't enjoyable.", ["likes", "dislikes"])
        ]


        feedback = {}
        
        # Randomly set children not to include feedback
        num_children = len(self.true_preferences.keys())
        children = list(self.true_preferences.keys())
        # Generated random indexes between 0 to n-1 and 0 - n long
        random_indices = self.generate_random_indices(num_children, average_exclude, std_dev_exclude)

        # Directly map the indices to children to exclude
        children_to_exclude = [children[idx] for idx in random_indices]

        for child, prefs in self.true_preferences.items():
            if child in children_to_exclude:
                continue
            
            available_ingredients = {
                "likes": prefs['known']['likes'] + prefs['unknown']['likes'],
                "neutral": prefs['known']['neutral'] + prefs['unknown']['neutral'],
                "dislikes": prefs['known']['dislikes'] + prefs['unknown']['dislikes'],
            }

            valid_comments = []

            for comment_template, feedback_types in comments:
                matched_ingredients = []
                used_ingredients = set()

                for feedback_type in feedback_types:
                    for category in available_ingredients:
                        if feedback_type in category:
                            possible_ingredients = [ingredient for ingredient in self.menu_plan if ingredient in available_ingredients[category] and ingredient not in used_ingredients]
                            if possible_ingredients:
                                chosen_ingredient = random.choice(possible_ingredients)
                                matched_ingredients.append(chosen_ingredient)
                                used_ingredients.add(chosen_ingredient)
                                break

                if len(matched_ingredients) == len(feedback_types):
                    valid_comments.append((comment_template, matched_ingredients, feedback_types))

            if valid_comments:
                comment_template, matched_ingredients, feedback_types = random.choice(valid_comments)
                comment = comment_template.format(*matched_ingredients)
                correct_action = {ingredient: feedback_types[idx] for idx, ingredient in enumerate(matched_ingredients)}
                feedback[child] = {"comment": comment, "correct_action": correct_action}

        return feedback

    def display_feedback_changes(self, original_preferences) -> None:
        """
        Display the changes made to children's preferences.
        """
        
        label_mapping = {'2': 'likes', '1': 'neutral', '0': 'dislikes'}
        for change in self.changes:
            print(f"\nChild {change['child']} had Action:")
            print(f"{change['ingredient']} {change['change']}.")
            true_preference_label = PreferenceModel.get_true_preference_label(original_preferences, change['ingredient'], change['child'])
            if label_mapping[str(true_preference_label)] == change['change'].split(" ")[-1]:
                pass
                print("Correct Action")
            else:
                print("Incorrect Preferences: should be", label_mapping[str(true_preference_label)])

    def display_incorrect_feedback_changes(self) -> None:
        """
        Display the incorrect comments and the reasons for incorrect predictions.
        """
        print("\n")
        for comment in self.incorrect_comments:
            print(f"Child {comment['child']} commented: '{comment['comment']}'")
            print(f"Predicted: {comment['predicted']}, Actual: {comment['actual']}\n")
