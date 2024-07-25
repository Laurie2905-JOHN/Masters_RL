import logging
import copy
from models.preferences.data_utils import get_child_data, initialize_child_preference_data
from utils.process_data import get_data
from models.preferences.prediction import PreferenceModel
from models.preferences.voting import IngredientNegotiator
from models.preferences.sentiment_analysis import SentimentAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(seed=4):
    ingredient_df = get_data("data.csv")
    child_feature_data = get_child_data()
    child_preference_data = initialize_child_preference_data(child_feature_data, ingredient_df, split=0.5, seed=seed, plot_graphs=False)

    # Make a deep copy of the child_preference_data to avoid modifying the original
    child_preference_data_copy = copy.deepcopy(child_preference_data)
    
    Predictor = PreferenceModel(ingredient_df, child_feature_data, child_preference_data_copy, apply_SMOTE=True, seed=seed)
    updated_known_and_predicted_preferences = Predictor.run_pipeline()

    # Negotiate the ingredients
    previous_feedback = {}
    previous_fairness_index = {}
    previous_utility = {}
    
    Negotiator = IngredientNegotiator(seed, ingredient_df, updated_known_and_predicted_preferences, previous_feedback, previous_fairness_index, previous_utility)
    
    # Simple
    negotiated_ingredient_order_simple, unavailable_ingredients_simple = Negotiator.negotiate_ingredients_simple()
    preference_score_function_simple = Negotiator.create_preference_score_function(negotiated_ingredient_order_simple)
    
    # Complex
    negotiated_ingredient_order_complex, unavailable_ingredients_complex = Negotiator.negotiate_ingredients_complex()
    preference_score_function_complex = Negotiator.create_preference_score_function(negotiated_ingredient_order_complex)
    
    Negotiator.close(log_file="Gini.json")
    
    # Feedback
    menu_plan = ['Wheat bread and rolls white (refined flour)', 'Potatoes', 'Sour cream plain', 'Bacon', 'Sweet corn', 'Cauliflowers']
    
    Sentiment = SentimentAnalyzer(child_preference_data_copy, menu_plan, seed=seed)
    
    changes, updated_preferences_with_feedback, accuracy, incorrect_comments, feedback_given = Sentiment.get_sentiment_and_update_data(updated_known_and_predicted_preferences, plot_confusion_matrix=False)

    Sentiment.display_feedback_changes(changes, child_preference_data)
    Sentiment.display_incorrect_feedback_changes(incorrect_comments)
    
    print("Sentiment Accuracy:", accuracy)
    
    print("Accuracy After Feedback:")
    Predictor.print_preference_difference_and_accuracy(child_preference_data_copy, updated_preferences_with_feedback, summary_only=True)

if __name__ == "__main__":
    main(seed=43)
