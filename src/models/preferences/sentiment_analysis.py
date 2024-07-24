from transformers import pipeline
import torch
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Function to analyze the sentiment of each sentence using BERT
def analyze_sentiment(comment, sentiment_analyzer):
    result = sentiment_analyzer(comment)
    label = result[0]['label']

    if label == 'POS':
        return 'likes'
    elif label == 'NEG':
        return 'dislikes'
    else:
        return 'neutral'

def get_sentiment_and_update_data(preferences, feedback, menu_plan, plot_confusion_matrix=False):
    changes = []
    incorrect_comments = []
    true_labels = []
    pred_labels = []
    
    # Mapping for confusion matrix
    label_mapping = {'likes': 2, 'neutral': 1, 'dislikes': 0}

    # Check if GPU is available and set device accordingly
    device = 0 if torch.cuda.is_available() else -1
    # Load the sentiment analysis pipeline with a specific model
    sentiment_analyzer = pipeline('sentiment-analysis', model="finiteautomata/bertweet-base-sentiment-analysis", device=device)
    
    # Iterate over each child's feedback
    for child, fb in feedback.items():
        # Split comments into sentences based on punctuation
        comments = re.split(r'[,.!?]', fb["comment"].lower())
        correct_action = fb["correct_action"]
        
        # Analyze each comment's sentiment
        for sentence in comments:
            if sentence.strip():  # Check if the sentence is not empty
                pred_label = analyze_sentiment(sentence.strip(), sentiment_analyzer)
                
                # Check for mentions of each ingredient in the sentence
                for ingredient in menu_plan:
                    if ingredient.lower() in sentence:
                        current_list = None
                        # Determine current list of the ingredient
                        if ingredient in preferences[child]["likes"]:
                            current_list = "likes"
                        elif ingredient in preferences[child]["dislikes"]:
                            current_list = "dislikes"
                        elif ingredient in preferences[child]["neutral"]:
                            current_list = "neutral"
                        
                        new_list = pred_label if pred_label in ["likes", "dislikes"] else "neutral"
                        
                        if current_list != new_list:
                            change = {"child": child, "ingredient": ingredient, "change": ""}
                            # Remove from current list
                            if current_list:
                                preferences[child][current_list].remove(ingredient)
                                change["change"] += f"removed from {current_list}"
                            
                            # Add to new list
                            preferences[child][new_list].append(ingredient)
                            if change["change"]:
                                change["change"] += ", "
                            change["change"] += f"added to {new_list}"
                            
                            changes.append(change)
                        
                        true_labels.append(correct_action[ingredient])
                        pred_labels.append(pred_label)
                        
                        # Check if the prediction was incorrect
                        if pred_label != correct_action[ingredient]:
                            incorrect_comments.append({
                                "child": child,
                                "comment": sentence,
                                "predicted": pred_label,
                                "actual": correct_action[ingredient]
                            })

    # Calculate accuracy
    correct_actions = sum(1 for true, pred in zip(true_labels, pred_labels) if true == pred)
    total_actions = len(true_labels)
    accuracy = correct_actions / total_actions if total_actions > 0 else 0

    # Plot confusion matrix if flag is set
    if plot_confusion_matrix and total_actions > 0:
        cm = confusion_matrix([label_mapping[label] for label in true_labels], 
                              [label_mapping[label] for label in pred_labels], 
                              labels=[0, 1, 2])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['likes', 'neutral', 'dislikes'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.show()

    return changes, preferences, accuracy, incorrect_comments

from models.preferences.prediction import get_true_preference_label

# Function to display the changes made to children's preferences
def display_feedback_changes(changes, child_preference_data):
    label_mapping = {'2': 'likes', '1': 'neutral', '0': 'dislikes'}
    for change in changes:
        print(f"\nChild {change['child']} had Action:")
        print(f"{change['ingredient']} {change['change']}.")
        true_preference_label = get_true_preference_label(change['ingredient'], child_preference_data, change['child'])
        print("Correct Preferences:", label_mapping[str(true_preference_label)])
        
# Function to display incorrect comments and reasons
def display_incorrect_feedback_changes(incorrect_comments):
    print("\n")
    for comment in incorrect_comments:
        print(f"Child {comment['child']} commented: '{comment['comment']}'")
        print(f"Predicted: {comment['predicted']}, Actual: {comment['actual']}\n")