import logging
from models.preferences.data_utils import get_child_data, initialize_child_preference_data, print_preference_difference_and_accuracy
from utils.process_data import get_data
from models.preferences.prediction import PreferenceModel
from models.preferences.voting import IngredientNegotiator
from models.preferences.sentiment_analysis import SentimentAnalyzer
from models.preferences.menu_generator import RandomMenuGenerator
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(weight_function: str, seed: Optional[int] = None, iterations: int = 10):
    ingredient_df = get_data("data.csv")
    child_feature_data = get_child_data()
    child_preference_data = initialize_child_preference_data(child_feature_data, ingredient_df, split=0.5, seed=seed, plot_graphs=False)
    
    # Initialize menu generator
    menu_generator = RandomMenuGenerator(seed=seed)
    
    # Initial Prediction of Preferences
    Predictor = PreferenceModel(ingredient_df, child_feature_data, child_preference_data, apply_SMOTE=True, seed=seed)
    updated_known_and_predicted_preferences = Predictor.run_pipeline()
    print_preference_difference_and_accuracy(child_preference_data, updated_known_and_predicted_preferences, summary_only=True)
    
    previous_feedback = {}
    previous_fairness_index = {}
    previous_utility = {}
    
    for i in range(iterations):
        logging.info(f"Iteration {i+1}")
        
        # Initial Negotiation of Ingredients
        Negotiator = IngredientNegotiator(seed, ingredient_df, updated_known_and_predicted_preferences, previous_feedback, previous_fairness_index, previous_utility)
        
        if weight_function == "simple":
            # Simple
            negotiated, unavailable = Negotiator.negotiate_ingredients_simple()
        elif weight_function == "complex":
            # Complex
            negotiated, unavailable = Negotiator.negotiate_ingredients_complex()
        else:
            raise ValueError("Invalid weight function")
        
        # Close negotiator to get initial gini stats on voting
        Negotiator.close(log_file=f"Gini_iteration_{i+1}.json")
        
        # Generate random menu
        menu_plan = menu_generator.generate_random_menu(negotiated, unavailable)
        
        # Sentiment analysis and get updated preference list
        Sentiment = SentimentAnalyzer(child_preference_data, menu_plan, seed=seed)
        updated_true_preferences_with_feedback, accuracy, feedback_given = Sentiment.get_sentiment_and_update_data(plot_confusion_matrix=False)

        logging.info(f"Sentiment Accuracy: {accuracy}")
        
        # Update predictor with new preferences
        Predictor = PreferenceModel(ingredient_df, child_feature_data, updated_true_preferences_with_feedback, apply_SMOTE=True, seed=seed)
        updated_known_and_predicted_preferences = Predictor.run_pipeline()
        
        print_preference_difference_and_accuracy(child_preference_data, updated_known_and_predicted_preferences, summary_only=True)
        
        previous_feedback = feedback_given
        previous_fairness_index = {}
        previous_utility = {}

if __name__ == "__main__":
    main(weight_function="simple", seed=None, iterations=10)
