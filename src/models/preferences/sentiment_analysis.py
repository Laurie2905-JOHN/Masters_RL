from transformers import pipeline
import torch
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from models.preferences.prediction import PreferenceModel
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Tuple, Union, Optional
import copy
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

class SentimentAnalyzer:
    def __init__(self, current_known_and_unknown_preferences, menu_plan, model_name: str = "finiteautomata/bertweet-base-sentiment-analysis", seed: int = 42):
        """
        Initialize the sentiment analyzer with the specified model.
        """
        self.current_known_and_unknown_preferences = copy.deepcopy(current_known_and_unknown_preferences)
        
        model_name_dict = {
                            'roberta': "cardiffnlp/twitter-roberta-base-sentiment",
                            'bertweet':  "finiteautomata/bertweet-base-sentiment-analysis",
                            'distilroberta':  "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
                            '5_star':  "nlptown/bert-base-multilingual-uncased-sentiment",
                            'perfect': "perfect",
                            'TextBlob': "TextBlob",
                            'Vader': "Vader"
                            }
        
        if model_name not in model_name_dict.keys():
            raise ValueError(f"Unknown model name: {model_name}")
        
        model = model_name_dict[model_name]
        
        self.model_name = model
        
        self.menu_plan = menu_plan
        self.label_mapping = {'likes': 2, 'neutral': 1, 'dislikes': 0}
        self.feedback = self.get_feedback()
        self.changes = []
        self.incorrect_comments = []
        self.is_star_model = "5_star" in model_name

        self.is_perfect_prediction = "perfect" in model_name

        if not self.is_perfect_prediction:
            if model_name == "TextBlob":
                self.sentiment_analyzer = TextBlob
            elif model_name == "Vader":
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            else:
                self.sentiment_analyzer = pipeline('sentiment-analysis', model=model, device='cpu')
            
    def analyze_sentiment(self, comment: str) -> str:
        """
        Analyze the sentiment of a given comment and return the corresponding label.
        """
        if self.model_name != "Vader":
            result = self.sentiment_analyzer(comment)
        
        if self.is_star_model:
            # For 5-star model
            star_rating = int(result[0]['label'].split()[0])
            if star_rating >= 4:
                return 'likes'
            elif star_rating <= 2:
                return 'dislikes'
            else:
                return 'neutral'
            
        elif self.model_name == "TextBlob":
            # For TextBlob model
            polarity = result.sentiment.polarity
            if polarity > 0.45:
                return 'likes'
            elif polarity < 0.2:
                return 'dislikes'
            else:
                return 'neutral'
            
        elif self.model_name == "Vader":
            
            vs = self.sentiment_analyzer.polarity_scores(comment)
            
            # Use the compound score to determine sentiment
            compound_score = vs['compound']

            # Classify based on the compound score
            if compound_score >= 0.4:
                return 'likes'
            elif compound_score <= 0.2:
                return 'dislikes'
            else:
                return 'neutral'
        else:
            # For 3-class sentiment model
            label = result[0]['label']
            if label in ['POS', 'LABEL_2', 'positive']:
                return 'likes'
            elif label in ['NEG', 'LABEL_0', 'negative']:
                return 'dislikes'
            elif label in ['NEU', 'LABEL_1', 'neutral']:
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
                    
                    if not self.is_perfect_prediction:
                        pred_label = self.analyze_sentiment(sentence.strip())
                        
                    for ingredient in self.menu_plan:
                        
                        if ingredient.lower() in sentence:
                            
                            if self.is_perfect_prediction:
                                pred_label = correct_action[ingredient]
                                
                            current_preference = self.get_current_preference_label(child, ingredient)
                            
                            if current_preference != pred_label:
                                change = self.update_preferences(child, ingredient, current_preference, pred_label)
                                self.changes.append(change)

                            try:
                                true_labels.append(correct_action[ingredient])
                            except KeyError:
                                true_labels.append(None)
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

        return self.current_known_and_unknown_preferences, accuracy, feedback, true_labels, pred_labels

    def get_current_preference_label(self, child: str, ingredient: str) -> Optional[str]:
        """
        Get the current preference list (likes, dislikes, neutral) or unknown.
        """ 
        if ingredient in self.current_known_and_unknown_preferences[child]['known']["likes"]:
            return "likes"
        elif ingredient in self.current_known_and_unknown_preferences[child]['known']["dislikes"]:
            return "dislikes"
        elif ingredient in self.current_known_and_unknown_preferences[child]['known']["neutral"]:
            return "neutral"
        elif ingredient in self.current_known_and_unknown_preferences[child]['unknown']["likes"] + self.current_known_and_unknown_preferences[child]['unknown']["dislikes"]  + self.current_known_and_unknown_preferences[child]['unknown']["neutral"]: 
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
        if current_preference == "unknown":
            for category in self.current_known_and_unknown_preferences[child]['unknown']:
                if ingredient in self.current_known_and_unknown_preferences[child]['unknown'][category]:
                    self.current_known_and_unknown_preferences[child]['unknown'][category].remove(ingredient)
                    self.current_known_and_unknown_preferences[child]['known'][new_preference].append(ingredient)
                    change["change"] += f"removed from {category} (unknown), added to {new_preference}"
                    break
        else:
            self.current_known_and_unknown_preferences[child]['known'][current_preference].remove(ingredient)
            self.current_known_and_unknown_preferences[child]['known'][new_preference].append(ingredient)
            change["change"] += f"removed from {current_preference}, added to {new_preference}"

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
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['dislikes', 'neutral', 'likes'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix for {self.model_name}')
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

    def sample_comments(self, comments: List[Tuple[str, List[str]]], likes_percent: int, neutral_percent: int, dislikes_percent: int, sample_size: int) -> List[Tuple[str, List[str]]]:
        """
        Sample comments based on the specified proportions for likes, neutral, and dislikes.

        Parameters:
        comments (List[Tuple[str, List[str]]]): The list of comment templates and their feedback types.
        likes_percent (int): The percentage of comments that should be 'likes'.
        neutral_percent (int): The percentage of comments that should be 'neutral'.
        dislikes_percent (int): The percentage of comments that should be 'dislikes'.
        sample_size (int): The total number of comments to sample.

        Returns:
        List[Tuple[str, List[str]]]: The sampled list of comments.
        """
        percent_sum = likes_percent + neutral_percent + dislikes_percent

        if percent_sum != 100:
            raise ValueError(f"The sum of likes_percent, neutral_percent, and dislikes_percent must equal 100. Equals {percent_sum}")
        
        likes_comments = [comment for comment in comments if 'likes' in comment[1]]
        neutral_comments = [comment for comment in comments if 'neutral' in comment[1]]
        dislikes_comments = [comment for comment in comments if 'dislikes' in comment[1]]

        likes_sample_size = int(sample_size * (likes_percent / 100))
        neutral_sample_size = int(sample_size * (neutral_percent / 100))
        dislikes_sample_size = int(sample_size * (dislikes_percent / 100))

        sampled_likes = self.sample_with_replacement(likes_comments, likes_sample_size)
        sampled_neutral = self.sample_with_replacement(neutral_comments, neutral_sample_size)
        sampled_dislikes = self.sample_with_replacement(dislikes_comments, dislikes_sample_size)

        # Combine sampled comments and fill the rest if sample sizes are less than required
        sampled_comments = sampled_likes + sampled_neutral + sampled_dislikes
        remaining_sample_size = sample_size - len(sampled_comments)

        if remaining_sample_size > 0:
            remaining_comments = self.sample_with_replacement(comments, remaining_sample_size)
            sampled_comments += remaining_comments

        # Shuffle the sampled comments to ensure randomness
        random.shuffle(sampled_comments)

        return sampled_comments

    def sample_with_replacement(self, population: List[Tuple[str, List[str]]], k: int) -> List[Tuple[str, List[str]]]:
        """
        Sample k elements from the population with replacement.

        Parameters:
        population (List): The population to sample from.
        k (int): The number of elements to sample.

        Returns:
        List: The sampled elements.
        """
        if len(population) == 0:
            raise ValueError("Population must contain at least one element to sample from.")
        return [random.choice(population) for _ in range(k)]

    def get_feedback(self, average_exclude: int = 5, std_dev_exclude: int = 2, seed: Optional[int] = None) -> Dict[str, Dict[str, Union[str, Dict[str, str]]]]:
        """
        Generate feedback based on current_known_and_unknown_preferences (preferences are the true initialized ones, no error) and menu plan,
        providing randomized comments on the ingredients. Exclude an average of average_exclude children with a standard deviation of std_dev_exclude.

        Parameters:
        average_exclude (int): The average number of children to exclude from feedback.
        std_dev_exclude (int): The standard deviation for the number of children to exclude.
        seed (Optional[int]): Seed for random number generators to ensure reproducibility.

        Returns:
        Dict[str, Dict[str, Union[str, Dict[str, str]]]]: A dictionary containing feedback comments and the correct actions for each child.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        comments = [
            ("Loved the {} and {}.", ["likes", "likes"]),
            ("Really liked the {} with {} and the {} was tasty.", ["likes", "likes", "likes"]),
            ("Enjoyed the {} and {}.", ["likes", "likes"]),
            ("Loved the {} and {}, but the {} was not liked.", ["likes", "likes", "dislikes"]),
            ("The {} and {} dish was great.", ["likes", "likes"]),
            ("Loved the {} and {}, but not the {}.", ["likes", "likes", "dislikes"]),
            ("The {} and {} were amazing, but didn't enjoy the {} much.", ["likes", "likes", "dislikes"]),
            ("Enjoyed the {} and {}, but was okay with the {}.", ["likes", "likes", "neutral"]),
            ("Enjoyed the {} and {}, but not the {}.", ["likes", "likes", "dislikes"]),
            ("Really liked the {}, but the {} was too strong.", ["likes", "dislikes"]),
            ("Enjoyed the {}, but the {} was too bland.", ["likes", "dislikes"]),
            ("Loved the {}, but didn't like the {}.", ["likes", "dislikes"]),
            ("The {} was good, but the {} was not to my taste.", ["likes", "dislikes"]),
            ("Enjoyed the {}, but the {} was too overpowering.", ["likes", "dislikes"]),
            ("The {} was delicious, but the {} wasn't enjoyable.", ["likes", "dislikes"]),
            ("Didn't like the {}, but the {} was good.", ["dislikes", "likes"]),
            ("Loved the {}, but the {} was not appealing.", ["likes", "dislikes"]),
            ("The {} was tasty, but the {} wasn't.", ["likes", "dislikes"]),
            ("Loved the {}.", ["likes"]),
            ("The {} was tasty.", ["likes"]),
            ("Enjoyed the {}.", ["likes"]),
            ("The {} was great.", ["likes"]),
            ("Loved the {} and the {}.", ["likes", "likes"]),
            ("Really liked the {}, {} and {}.", ["likes", "likes", "likes"]),
            ("Enjoyed the {}, but the {} was not my favorite.", ["likes", "dislikes"]),
            ("Loved the {}, but the {} was just okay.", ["likes", "neutral"]),
            ("The {} was fantastic, but the {} was not as good.", ["likes", "dislikes"]),
            ("Really liked the {} and {}.", ["likes", "likes"]),
            ("Enjoyed the {} and the {}.", ["likes", "likes"]),
            ("Loved the {} and {}, but the {} was just okay.", ["likes", "neutral"]),
            ("The {} was great, but the {} was just okay.", ["likes", "neutral"]),
            ("Really liked the {}, but the {} was just okay.", ["likes", "neutral"]),
            ("Enjoyed the {}, but the {} was fine.", ["likes", "neutral"]),
            ("The {} was good, and the {} was okay.", ["likes", "neutral"]),
            ("Enjoyed the {} and {}, but the {} was okay.", ["likes", "likes", "neutral"]),
            ("The {} and {} were good, but the {} was just okay.", ["likes", "likes", "neutral"]),
            ("Loved the {}, but the {} was just okay.", ["likes", "neutral"]),
            ("Enjoyed the {}, but the {} was okay.", ["likes", "neutral"]),
            ("The {} was fantastic, but the {} was okay.", ["likes", "neutral"]),
            ("Loved the {} and {}, but the {} was just okay.", ["likes", "neutral"]),
            ("The {} was great, and the {} was okay.", ["likes", "neutral"]),
            ("Loved the {} and {}, but the {} was not as good.", ["likes", "neutral"]),
            ("Really liked the {}, but the {} was okay.", ["likes", "neutral"]),
            ("The {} was fine, but the {} wasn't good.", ["neutral", "dislikes"]),
            ("Didn't enjoy the {} much, but the {} was okay.", ["dislikes", "neutral"]),
            ("Didn't enjoy the {}, but the {} was okay.", ["dislikes", "neutral"]),
            ("Didn't like the {} in the dish, but the {} was fine.", ["dislikes", "neutral"]),
            ("The {} was okay, but the {} wasn't appealing.", ["neutral", "dislikes"]),
            ("Enjoyed the {} and {}, but the {} was okay.", ["likes", "likes", "neutral"]),
            ("Didn't like the {}, but the {} was okay.", ["dislikes", "neutral"]),
            ("The {} was fine, but the {} needed more flavor.", ["neutral", "dislikes"]),
            ("Enjoyed the {}, but the {} was just okay.", ["likes", "neutral"]),
            ("The {} was fine, and the {} was just okay.", ["neutral", "neutral"]),
            ("Didn't enjoy the {}, but the {} was okay.", ["dislikes", "neutral"]),
            ("The {} was okay, and the {} was fine.", ["neutral", "neutral"]),
            ("The {} was just okay, but the {} was good.", ["neutral", "likes"]),
            ("Did not enjoy the {} and {}.", ["dislikes", "dislikes"]),
            ("Didn't like the {} and {}.", ["dislikes", "dislikes"]),
            ("Didn't like the {} and {} in the dish.", ["dislikes", "dislikes"]),
            ("Didn't like the {}.", ["dislikes"]),
            ("Didn't like the {} in the dish.", ["dislikes"]),
            ("Didn't like the {}, but the {} was fine.", ["dislikes", "neutral"]),
            ("Didn't enjoy the {}.", ["dislikes"]),
            ("Didn't like the {} at all.", ["dislikes"]),
            ("Didn't enjoy the {}, and the {} was bad.", ["dislikes", "dislikes"]),
        ]

        feedback = {}

        # Randomly exclude a certain number of children from the feedback process
        num_children = len(self.current_known_and_unknown_preferences.keys())
        children = list(self.current_known_and_unknown_preferences.keys())
        # Generate random indices for children to exclude based on the average and standard deviation provided
        random_indices = self.generate_random_indices(num_children, average_exclude, std_dev_exclude)

        # Map the generated indices to the actual children to exclude
        children_to_exclude = [children[idx] for idx in random_indices]

        # Generate feedback for each child except those that are excluded
        for child, prefs in self.current_known_and_unknown_preferences.items():
            if child in children_to_exclude:
                continue

            # Aggregate known and unknown preferences for each feedback type (likes, neutral, dislikes)
            available_ingredients = {
                "likes": prefs['known']['likes'] + prefs['unknown']['likes'],
                "neutral": prefs['known']['neutral'] + prefs['unknown']['neutral'],
                "dislikes": prefs['known']['dislikes'] + prefs['unknown']['dislikes'],
            }

            valid_comments = []

            # Sample comments based on specified proportions for likes, neutral, and dislikes
            for comment_template, feedback_types in self.sample_comments(comments, likes_percent=40, neutral_percent=20, dislikes_percent=40, sample_size=len(comments)):
                matched_ingredients = []
                used_ingredients = set()

                # Match ingredients to the feedback types specified in the comment template
                for feedback_type in feedback_types:
                    for category in available_ingredients:
                        if feedback_type in category:
                            possible_ingredients = [ingredient for ingredient in self.menu_plan if ingredient in available_ingredients[category] and ingredient not in used_ingredients]
                            if possible_ingredients:
                                chosen_ingredient = random.choice(possible_ingredients)
                                matched_ingredients.append(chosen_ingredient)
                                used_ingredients.add(chosen_ingredient)
                                break

                # Ensure that all required ingredients are matched before adding the comment to valid comments
                if len(matched_ingredients) == len(feedback_types):
                    valid_comments.append((comment_template, matched_ingredients, feedback_types))

            # Filter out comments that do not match the number of ingredients
            valid_comments = [comment for comment in valid_comments if comment[0].count("{}") == len(comment[1])]

            # Select a random valid comment and generate the feedback entry for the child
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

