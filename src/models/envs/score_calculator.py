class BaseScoreCalculator:
    def __init__(self, main_instance):
        """Initialize with a reference to the main instance."""
        self.main = main_instance

    def calculate_scores(self):
        """Calculate all scores and identify which targets are not met."""
        nutrient_score, nutrient_target_met = self._calculate_nutrient_score()
        cost_score, cost_target_met = self._calculate_cost_score(nutrient_target_met)
        co2_score, co2_target_met = self._calculate_co2_score(nutrient_target_met)

        # Collect all scores
        scores = [nutrient_score, cost_score, co2_score]
        
        # Collect unmet targets
        targets = []
        if not nutrient_target_met:
            targets.append('nutrient')
        if not cost_target_met:
            targets.append('cost')
        if not co2_target_met:
            targets.append('co2')

        return scores, targets

    def calculate_score(self, items, target, normalize=True):
        """Calculate a generic score and check if the target is met."""
        score = 0
        target_met = True  # Assume target is met until proven otherwise
        norm_factor = len(items)

        for key, value in items.items():
            if self.is_within_target(key, value, target):
                score += 1
            else:
                target_met = False
                # Calculate score based on distance from target
                min_target, max_target = target[key]
                distance = min(abs(min_target - value), abs(value - max_target))
                max_distance = max(abs(min_target), abs(max_target))
                score += 1 - (distance / max_distance)

        if normalize:
            score = score / norm_factor

        return score, target_met

    def is_within_target(self, key, value, target):
        """Check if the value is within the target range."""
        min_target, max_target = target[key]
        return min_target <= value <= max_target
    
    def _check_preference_target(self, preference_score):
        """Check if the preference target is met."""
        return preference_score >= self.main.preference_target
    
    def _calculate_preference_score(self):
        """Calculate the preference score"""
        
        preference_score = 0
        
        current_meal_plan, _, _ = self.main.get_current_meal_plan()
        
        for ingredient in current_meal_plan.keys():
            preference_score += self.main.preference_score_function(ingredient)
        
        norm_factor = len(current_meal_plan.keys())
        
        preference_score = preference_score / norm_factor
        
        # Check if the preference target is met
        preference_target_met = self._check_preference_target(self, preference_score)
        
        return preference_score, preference_target_met
    

class ScoreCalculatorSparse(BaseScoreCalculator):
    def __init__(self, main_instance):
        """Initialize ScoreCalculatorSparse with a reference to the main instance."""
        super().__init__(main_instance)

    def _calculate_nutrient_score(self):
        """Calculate the nutrient score and check if the target is met."""
        return self.calculate_score(self.main.nutrient_averages, self.main.nutrient_target_ranges)

    def _calculate_cost_score(self):
        """Calculate the cost score and check if the target is met."""
        cost_target = {key: (0, self.main.target_cost_per_meal) for key in self.main.menu_cost.keys()}
        return self.calculate_score(self.main.menu_cost, cost_target, normalize=False)

    def _calculate_co2_score(self):
        """Calculate the CO2 score and check if the target is met."""
        co2_target = {key: (0, self.main.target_co2_g_per_meal) for key in self.main.co2_g.keys()}
        return self.calculate_score(self.main.co2_g, co2_target, normalize=False)

    def calculate_scores(self):
        """Calculate all scores and return binary values for target met status."""
        _, nutrient_target_met = self._calculate_nutrient_score()
        _, cost_target_met = self._calculate_cost_score()
        _, co2_target_met = self._calculate_co2_score()
        _, preference_target_met = self._calculate_preference_score()
        
        # Collect binary scores based on target met status
        scores = [
            1 if nutrient_target_met else 0,
            1 if cost_target_met else 0,
            1 if co2_target_met else 0,
            1 if preference_target_met else 0
        ]

        return scores


class ScoreCalculatorShaped(BaseScoreCalculator):
    def __init__(self, main_instance):
        """Initialize ScoreCalculator with a reference to the main instance."""
        super().__init__(main_instance)

    def _calculate_nutrient_score(self, group_target_met):
        """Calculate the nutrient score and check if the target is met."""
        
        if not group_target_met:
            return 0, False  # No cost score if nutrient target is met
        
        return self.calculate_score(self.main.nutrient_averages, self.main.nutrient_target_ranges)

    def _calculate_cost_score(self, group_target_met):
        """Calculate the cost score and check if the target is met."""
        if not group_target_met:
            return 0, False  # No cost score if nutrient target is met

        cost_target = {key: (0, self.main.target_cost_per_meal) for key in self.main.menu_cost.keys()}
        return self.calculate_score(self.main.menu_cost, cost_target, normalize=False)
    
    def _calculate_co2_score(self, group_target_met):
        """Calculate the CO2 score and check if the target is met."""
        if not group_target_met:
            return 0, False  # No CO2 score if nutrient target is not met

        co2_target = {key: (0, self.main.target_co2_g_per_meal) for key in self.main.co2_g.keys()}
        return self.calculate_score(self.main.co2_g, co2_target, normalize=False)
    
    def _calculate_group_score(self):
        """Calculate the group score and check if the target is met."""

        group_target = self.main.ingredient_group_portion_targets
        scores, target_met = 0, True
        norm_factor = len(self.main.ingredient_group_count.keys())

        for key, value in self.main.ingredient_group_count.items():
            portion = (
                self.main.ingredient_group_portion[key] / value
                if value != 0 else 0
            )
            min_target, max_target = group_target[key]

            if min_target <= portion <= max_target:
                scores += 1
            else:
                target_met = False
                # Calculate score based on distance from target
                distance = min(abs(min_target - portion), abs(portion - max_target))
                max_distance = max(abs(min_target), abs(max_target))
                if max_distance == 0:
                    scores += 1
                scores += (1 - (distance / max_distance))

        return scores / norm_factor, target_met

    def is_within_portion_range(self, group: str):
        """Check if the portion is within the target range."""
        portion = (
            self.main.ingredient_group_portion[group] / self.main.ingredient_group_count[group]
            if self.main.ingredient_group_count[group] != 0 else 0
        )
        min_target, max_target = self.main.ingredient_group_portion_targets[group]
        terminate = portion > 300
        return min_target <= portion <= max_target, terminate

    def calculate_scores(self):
        """Calculate all scores and identify which targets are not met."""
        _, group_target_met = self._calculate_group_score()
        nutrient_score, nutrient_target_met = self._calculate_nutrient_score(group_target_met)
        cost_score, cost_target_met = self._calculate_cost_score(group_target_met)
        co2_score, co2_target_met = self._calculate_co2_score(group_target_met)
        preference_score, preference_target_met = self._calculate_preference_score()
        # Collect all scores
        scores = [nutrient_score, cost_score, co2_score, preference_score]
        
        # Collect unmet targets
        targets = []
        if not nutrient_target_met:
            targets.append('nutrient')
        if not cost_target_met:
            targets.append('cost')
        if not co2_target_met:
            targets.append('co2')
        if not preference_target_met:
            targets.append('preference score')
            
        return scores, targets


