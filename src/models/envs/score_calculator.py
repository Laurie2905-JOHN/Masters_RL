class ScoreCalculator:
    def __init__(self, main_instance):
        self.main = main_instance
    
    def calculate_scores(self):
        group_score, group_target_met = self._calculate_group_score()
        nutrient_score, nutrient_target_met = self._calculate_nutrient_score()
        cost_score, cost_target_met = self._calculate_cost_score()
        co2_score, co2_target_met = self._calculate_co2_score()

        scores = [nutrient_score, group_score, cost_score, co2_score]
        
        targets = []
        if not nutrient_target_met:
            targets.append('nutrient')
        if not group_target_met:
            targets.append('group')
        if not cost_target_met:
            targets.append('cost')
        if not co2_target_met:
            targets.append('co2')

        return scores, targets

    def _calculate_nutrient_score(self):
        nutrient_score = 0  
        nutrient_target_met = False
        if self.main.step_to_reward:
            nutrient_target_met = True
            
            norm_factor = len(self.main.nutrient_averages.keys())
            for key, value in self.main.nutrient_averages.items():
                target_min, target_max = self.main.nutrient_target_ranges[key]
                if target_min <= value <= target_max:
                    nutrient_score += 1 / norm_factor
                else:
                    nutrient_target_met = False

            nutrient_score = nutrient_score / norm_factor
            
        return nutrient_score, nutrient_target_met

    def _calculate_cost_score(self):
        
        cost_score = 0
        cost_target_met = False
        if self.main.step_to_reward:
            cost_target_met = True
            norm_factor = len(self.main.menu_cost.keys())
            for key, value in self.main.menu_cost.items():
                if value <= self.main.target_cost_per_meal:
                    cost_score += 1
                else:
                    cost_target_met = False

            cost_score = cost_score / norm_factor
        return cost_score, cost_target_met

    def _calculate_co2_score(self):
        
        co2_score = 0
        co2_target_met = False
        if self.main.step_to_reward:
            co2_target_met = True
            norm_factor = len(self.main.co2_g.keys())
            for key, value in self.main.co2_g.items():
                if value <= self.main.target_co2_g_per_meal:
                    co2_score += 1
                else:
                    co2_target_met = False

            co2_score = co2_score / norm_factor
                
        return co2_score, co2_target_met

    def _calculate_group_score(self):
        group_score = 0
        group_target_met = False
        if self.main.step_to_reward:
            group_target_met = True
            norm_factor = len(self.main.ingredient_group_count.keys())
            for key, value in self.main.ingredient_group_count.items():
                portion_target, _ = self.is_within_portion_range(key)

                if value == self.main.ingredient_group_count_targets[key] and portion_target:
                    group_score += 1
                else:
                    group_target_met = False

            group_score = group_score / norm_factor
        return group_score, group_target_met

    def is_within_portion_range(self, group: str):
        portion = (
            self.main.ingredient_group_portion[group] / self.main.ingredient_group_count[group]
            if self.main.ingredient_group_count[group] != 0 else 0
        )
        min_target, max_target = self.main.ingredient_group_portion_targets[group]
        terminate = portion > 300
        return min_target <= portion <= max_target, terminate


class ScoreCalculatorShaped:
    def __init__(self, main_instance):
        self.main = main_instance
    
    def calculate_scores(self):
        group_score, group_target_met = self._calculate_group_score()
        nutrient_score, nutrient_target_met = self._calculate_nutrient_score()
        cost_score, cost_target_met = self._calculate_cost_score()
        co2_score, co2_target_met = self._calculate_co2_score()

        scores = [nutrient_score, group_score, cost_score, co2_score]
        
        targets = []
        if not nutrient_target_met:
            targets.append('nutrient')
        if not group_target_met:
            targets.append('group')
        if not cost_target_met:
            targets.append('cost')
        if not co2_target_met:
            targets.append('co2')

        return scores, targets

    def _calculate_nutrient_score(self):
        nutrient_score = 0  
        nutrient_target_met = True  # Assume target is met until proven otherwise
        if self.main.step_to_reward:
            norm_factor = len(self.main.nutrient_averages.keys())
            for key, value in self.main.nutrient_averages.items():
                target_min, target_max = self.main.nutrient_target_ranges[key]
                if target_min <= value <= target_max:
                    nutrient_score += 1
                else:
                    nutrient_target_met = False
                    distance = min(abs(target_min - value), abs(value - target_max))
                    max_distance = max(abs(target_min), abs(target_max))
                    nutrient_score += (1 - (distance / max_distance))

            nutrient_score = nutrient_score / norm_factor
            
        return nutrient_score, nutrient_target_met

    def _calculate_cost_score(self):
        cost_score = 0
        cost_target_met = True  # Assume target is met until proven otherwise
        if self.main.step_to_reward:
            norm_factor = len(self.main.menu_cost.keys())
            for key, value in self.main.menu_cost.items():
                if value <= self.main.target_cost_per_meal:
                    cost_score += 1
                else:
                    cost_target_met = False
                    distance = abs(self.main.target_cost_per_meal - value)
                    max_distance = self.main.target_cost_per_meal
                    cost_score += 1 - (distance / max_distance)

            cost_score = cost_score / norm_factor
        return cost_score, cost_target_met

    def _calculate_co2_score(self):
        co2_score = 0
        co2_target_met = True  # Assume target is met until proven otherwise
        if self.main.step_to_reward:
            norm_factor = len(self.main.co2_g.keys())
            for key, value in self.main.co2_g.items():
                if value <= self.main.target_co2_g_per_meal:
                    co2_score += 1
                else:
                    co2_target_met = False
                    distance = abs(self.main.target_co2_g_per_meal - value)
                    max_distance = self.main.target_co2_g_per_meal
                    co2_score += 1 - (distance / max_distance)

            co2_score = co2_score / norm_factor
                
        return co2_score, co2_target_met

    def _calculate_group_score(self):
        group_score = 0
        group_target_met = True  # Assume target is met until proven otherwise
        if self.main.step_to_reward:
            norm_factor = len(self.main.ingredient_group_count.keys())
            for key, value in self.main.ingredient_group_count.items():
                portion = (
                    self.main.ingredient_group_portion[key] / value
                    if value != 0 else 0
                )
                min_target, max_target = self.main.ingredient_group_portion_targets[key]
                
                if min_target <= portion <= max_target:
                    group_score += 1
                else:
                    group_target_met = False
                    distance = min(abs(min_target - portion), abs(portion - max_target))
                    max_distance = max(abs(min_target), abs(max_target))
                    group_score += (1 - (distance / max_distance))

            group_score = group_score / norm_factor
        return group_score, group_target_met

    def is_within_portion_range(self, group: str):
        portion = (
            self.main.ingredient_group_portion[group] / self.main.ingredient_group_count[group]
            if self.main.ingredient_group_count[group] != 0 else 0
        )
        min_target, max_target = self.main.ingredient_group_portion_targets[group]
        terminate = portion > 300
        return min_target <= portion <= max_target, terminate
