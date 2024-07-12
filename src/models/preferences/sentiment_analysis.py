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

def extract_preferences_and_update_data(preferences, feedback, ingredient_list, plot_confusion_matrix=False):
    changes = []
    incorrect_comments = []
    true_labels = []
    pred_labels = []
    
    # Mapping for confusion matrix
    label_mapping = {'likes': 0, 'neutral': 1, 'dislikes': 2}

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
                for ingredient in ingredient_list:
                    if ingredient.lower() in sentence:
                        change = {"child": child, "ingredient": ingredient, "change": ""}
                        
                        # Determine the appropriate category based on polarity
                        if pred_label == 'likes':  
                            if ingredient not in preferences[child]['known']["likes"]:
                                preferences[child]['known']["likes"].append(ingredient)
                                change["change"] = "added to likes"
                        elif pred_label == 'dislikes':
                            if ingredient not in preferences[child]['known']["dislikes"]:
                                preferences[child]['known']["dislikes"].append(ingredient)
                                change["change"] = "added to dislikes"
                        else:
                            pred_label = 'neutral'
                            if ingredient not in preferences[child]['known']["neutral"]:
                                preferences[child]['known']["neutral"].append(ingredient)
                                change["change"] = "added to neutral"
                        
                        # Remove ingredient from other lists
                        if change["change"]:
                            if change["change"] != "added to likes" and ingredient in preferences[child]['known']["likes"]:
                                preferences[child]['known']["likes"].remove(ingredient)
                                change["change"] += ", removed from likes"
                            if change["change"] != "added to dislikes" and ingredient in preferences[child]['known']["dislikes"]:
                                preferences[child]['known']['dislikes'].remove(ingredient)
                                change["change"] += ", removed from dislikes"
                            if change["change"] != "added to neutral" and ingredient in preferences[child]['known']["neutral"]:
                                preferences[child]['known']["neutral"].remove(ingredient)
                                change["change"] += ", removed from neutral"
                            
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


# Function to display the changes made to children's preferences
def display_changes(changes):
    for change in changes:
        print("Action Taken:\n")
        print(f"Child {change['child']} had {change['ingredient']} {change['change']}.")
        
# Function to display incorrect comments and reasons
def display_incorrect_comments(incorrect_comments):
    for comment in incorrect_comments:
        print(f"Child {comment['child']} commented: '{comment['comment']}'")
        print(f"Predicted: {comment['predicted']}, Actual: {comment['actual']}\n")