from transformers import pipeline
import torch
import re

# Function to analyze the sentiment of each sentence using BERT
def analyze_sentiment(sentence, sentiment_analyzer):
    result = sentiment_analyzer(sentence)
    label = result[0]['label']
    score = result[0]['score']
    
    # Determine the polarity based on the sentiment label
    if label == 'POSITIVE':
        polarity = score
    else:
        polarity = -score
    return polarity


# Function to update children's data with new preferences based on feedback
def extract_preferences_and_update_data(preferences, feedback, menu_plan):
    
    # Check if GPU is available and set device accordingly
    device = 0 if torch.cuda.is_available() else -1

    # Load the sentiment analysis pipeline with a specific model
    sentiment_analyzer = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english", device=device)

    changes = []
    neutral_threshold = 0.1  # Threshold for considering an ingredient neutral
    
    # Iterate over each child's feedback
    for child, fb in feedback.items():
        # Split comments into sentences based on punctuation
        comments = re.split(r'[,.!?]', fb["comments"].lower())
        
        # Analyze each comment's sentiment
        for sentence in comments:
            if sentence.strip():  # Check if the sentence is not empty
                polarity = analyze_sentiment(sentence.strip(), sentiment_analyzer)
                
                # Check for mentions of each ingredient in the sentence
                for ingredient in menu_plan:
                    if ingredient in sentence:
                        change = {"child": child, "ingredient": ingredient, "change": ""}
                        
                        # Determine the appropriate category based on polarity
                        if polarity > neutral_threshold:
                            if ingredient not in preferences[child]["likes"]:
                                preferences[child]["likes"].append(ingredient)
                                change["change"] = "added to likes"
                        elif polarity < -neutral_threshold:
                            if ingredient not in preferences[child]["dislikes"]:
                                preferences[child]["dislikes"].append(ingredient)
                                change["change"] = "added to dislikes"
                        else:
                            if ingredient not in preferences[child]["neutral"]:
                                preferences[child]["neutral"].append(ingredient)
                                change["change"] = "added to neutral"
                        
                        # Remove ingredient from other lists
                        if change["change"]:
                            if change["change"] != "added to likes" and ingredient in preferences[child]["likes"]:
                                preferences[child]["likes"].remove(ingredient)
                                change["change"] += ", removed from likes"
                            if change["change"] != "added to dislikes" and ingredient in preferences[child]["dislikes"]:
                                preferences[child]["dislikes"].remove(ingredient)
                                change["change"] += ", removed from dislikes"
                            if change["change"] != "added to neutral" and ingredient in preferences[child]["neutral"]:
                                preferences[child]["neutral"].remove(ingredient)
                                change["change"] += ", removed from neutral"
                            
                            changes.append(change)
    
    return changes, preferences

# Function to display the changes made to children's preferences
def display_changes(changes):
    for change in changes:
        print(f"Child {change['child']} had {change['ingredient']} {change['change']}.")