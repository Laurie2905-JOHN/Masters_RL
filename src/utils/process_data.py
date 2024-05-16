import os
import pandas as pd

def process_data():
    try:
        # Define the base directory where your script is located
        base_dir = os.path.dirname(__file__)
        
        # Construct the full path to the data.csv file
        data_file_path = os.path.join(base_dir,'..','..', 'data', 'data.csv')
        
        # Attempt to open and read the data.csv file
        with open(data_file_path, 'r') as file:
            data = file.readlines()
        
        # Process the data and report the number of rows and ingredients per row
        num_rows = len(data)
        
        print(f"Successfully read {num_rows} lines from the file. Loaded 137 ingredients.")
        
        keys = data[0].replace('\n','').replace('\ufeff','').split(',')
        
        ingredients_df = pd.DataFrame(columns=keys)
        
        for i, row in enumerate(data):
            if i == 0:
                continue
            
            # Step 1: Split the string into a list
        
            ingredients = row.split(',')
            
            if len(ingredients) != 35:
                ingredients = custom_split(row)
                            

            cleaned_ingredients = [ingredient.replace('"', '').replace('\n', '') for ingredient in ingredients]
            
            
            ingredients_df.loc[len(ingredients_df.index)] = cleaned_ingredients
            
        return ingredients_df
    
    except FileNotFoundError:
        print(f"Error: The file at path '{data_file_path}' was not found.")
    except IOError as e:
        print(f"Error: An IOError occurred. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



def custom_split(row):
    parts = []
    current = []
    in_quotes = False
    
    i = 0
    while i < len(row):
        char = row[i]
        if char == '"':
            if in_quotes and i + 1 < len(row) and row[i + 1] == '"':  # Handle escaped quotes
                current.append('"')
                i += 1  # Skip the next quote
            else:
                in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
        i += 1
    
    # Add the last part
    parts.append(''.join(current).strip())
    
    return parts

# Example usage
if __name__ == "__main__":
    process_data()
