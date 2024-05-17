
import gym
from gym import spaces
import numpy as np

# Define the environment
class SimpleCalorieOnlyEnv(gym.Env):
    def __init__(self, ingredient_df, num_people):
        super(SimpleCalorieOnlyEnv, self).__init__()
        
        self.ingredient_df = ingredient_df
        
        self.num_people = num_people
        
        self.daily_selections = []
        
        # Define action and observation space
        n_ingredients = len(self.ingredient_df)
        
        self.action_space = spaces.Dict({
            'selection': spaces.MultiBinary(n_ingredients),
            'quantity': spaces.Box(low=0, high=100, shape=(n_ingredients,), dtype=np.float32)
        })
        
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        
        self.state = np.zeros(1, dtype=np.float32)

    def reset(self):
        self.daily_selections = []
        self.state = np.zeros(1, dtype=np.float32)
        return self.state
    
    def calculate_reward(self, action):
        
        reward = 0
        
        # Reward based on the number of selected ingredients
        total_selection = np.sum(action['selection'])
        
        if total_selection > 10:
            reward -= 10
        elif total_selection < 5:
            reward -= 10
        else:
            reward += 10
            
        calories_selected_ingredients = action['selection'] * action['quantity'] * self.ingredient_df['Calories_kcal_per_100g'].values
        
        # Calculate average calories per day per person
        average_calories_per_day = sum(calories_selected_ingredients) / self.num_people
        
        # Reward based on the average calories per day
        if 2000 <= average_calories_per_day <= 3000:
            reward += 100
            done = True
        else:
            done = False
            reward -= 10
        
        return reward, average_calories_per_day, done

    def step(self, action):
    
        # Calculate reward
        reward, average_calories_per_day, done = self.calculate_reward(action)
        
        info = {
            'average_calories_per_day': average_calories_per_day
        }

        return self.state, reward, done, info