import unittest

# Assuming the function and necessary variables are defined in a module named 'reward_module'
from models.reward.reward import group_count_reward

class TestGroupCountReward(unittest.TestCase):

    def setUp(self):
        # Setting up default values for the tests
        self.ingredient_group_portion = {
            'non_processed_protein': 300,
            'processed_protein': 200,
            'confectionary': 150,
            'vegetables': 500,
            'fruits': 300
        }
        
        self.ingredient_group_count = {
            'non_processed_protein': 3,
            'processed_protein': 2,
            'confectionary': 1,
            'vegetables': 5,
            'fruits': 3
        }
        
        self.ingredient_group_portion_targets = {
            'non_processed_protein': (100, 150),
            'processed_protein': (80, 120),
            'confectionary': (50, 60),
            'vegetables': (80, 100),
            'fruits': (90, 110)
        }
        
        self.ingredient_group_count_targets = {
            'non_processed_protein': 3,
            'processed_protein': 2,
            'confectionary': 1,
            'vegetables': 5,
            'fruits': 3
        }

    def test_all_targets_met(self):
        ingredient_group_count_rewards = {group: 0 for group in self.ingredient_group_count}
        
        expected_rewards = {
            'non_processed_protein': 50,
            'processed_protein': 50,
            'confectionary': 100,
            'vegetables': 100,
            'fruits': 100
        }
        
        rewards, all_targets_met = group_count_reward(ingredient_group_count_rewards)
        
        self.assertEqual(rewards, expected_rewards)
        self.assertTrue(all_targets_met)
    
    def test_protein_target_not_met(self):
        self.ingredient_group_count['non_processed_protein'] = 2
        
        ingredient_group_count_rewards = {group: 0 for group in self.ingredient_group_count}
        
        expected_rewards = {
            'non_processed_protein': -50,
            'processed_protein': -50,
            'confectionary': 100,
            'vegetables': 100,
            'fruits': 100
        }
        
        rewards, all_targets_met = group_count_reward(ingredient_group_count_rewards)
        
        self.assertEqual(rewards, expected_rewards)
        self.assertFalse(all_targets_met)
    
    def test_protein_target_exceeded(self):
        self.ingredient_group_count['non_processed_protein'] = 4
        
        ingredient_group_count_rewards = {group: 0 for group in self.ingredient_group_count}
        
        expected_rewards = {
            'non_processed_protein': -25,
            'processed_protein': -25,
            'confectionary': 100,
            'vegetables': 100,
            'fruits': 100
        }
        
        rewards, all_targets_met = group_count_reward(ingredient_group_count_rewards)
        
        self.assertEqual(rewards, expected_rewards)
        self.assertFalse(all_targets_met)
    
    def test_confectionary_target_not_met(self):
        self.ingredient_group_count['confectionary'] = 0
        
        ingredient_group_count_rewards = {group: 0 for group in self.ingredient_group_count}
        
        expected_rewards = {
            'non_processed_protein': 50,
            'processed_protein': 50,
            'confectionary': -100,
            'vegetables': 100,
            'fruits': 100
        }
        
        rewards, all_targets_met = group_count_reward(ingredient_group_count_rewards)
        
        self.assertEqual(rewards, expected_rewards)
        self.assertFalse(all_targets_met)
    
    def test_other_group_target_not_met(self):
        self.ingredient_group_count['vegetables'] = 4
        
        ingredient_group_count_rewards = {group: 0 for group in self.ingredient_group_count}
        
        expected_rewards = {
            'non_processed_protein': 50,
            'processed_protein': 50,
            'confectionary': 100,
            'vegetables': -100,
            'fruits': 100
        }
        
        rewards, all_targets_met = group_count_reward(ingredient_group_count_rewards)
        
        self.assertEqual(rewards, expected_rewards)
        self.assertFalse(all_targets_met)
    
    def test_other_group_target_exceeded(self):
        self.ingredient_group_count['vegetables'] = 6
        
        ingredient_group_count_rewards = {group: 0 for group in self.ingredient_group_count}
        
        expected_rewards = {
            'non_processed_protein': 50,
            'processed_protein': 50,
            'confectionary': 100,
            'vegetables': 0,
            'fruits': 100
        }
        
        rewards, all_targets_met = group_count_reward(ingredient_group_count_rewards)
        
        self.assertEqual(rewards, expected_rewards)
        self.assertFalse(all_targets_met)

    def test_combination_of_targets(self):
        self.ingredient_group_count['non_processed_protein'] = 4
        self.ingredient_group_count['processed_protein'] = 3
        self.ingredient_group_count['confectionary'] = 2
        self.ingredient_group_count['vegetables'] = 4
        self.ingredient_group_count['fruits'] = 2
        
        ingredient_group_count_rewards = {group: 0 for group in self.ingredient_group_count}
        
        expected_rewards = {
            'non_processed_protein': -50,
            'processed_protein': -50,
            'confectionary': -100,
            'vegetables': -100,
            'fruits': -100
        }
        
        rewards, all_targets_met = group_count_reward(ingredient_group_count_rewards)
        
        self.assertEqual(rewards, expected_rewards)
        self.assertFalse(all_targets_met)

if __name__ == '__main__':
    unittest.main()
