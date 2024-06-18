import os
import pandas as pd

def get_data(filename='small_data.csv'):
    try:
        # Define the base directory where your script is located
        base_dir = os.path.dirname(__file__)
        
        # Construct the full path to the data.csv file
        data_file_path = os.path.join(base_dir,'..','..', 'data', f'{filename}')
        
        # Attempt to open and read the data.csv file
        with open(data_file_path, 'r') as file:
            data = file.readlines()
        
        # Process the data and report the number of rows and ingredients per row
        num_rows = len(data) - 1
        
        keys = data[0].replace('\n','').replace('\ufeff','').split(',')
        
        ingredients_df = pd.DataFrame(columns=keys)
        
        for i, row in enumerate(data):
            if i == 0:
                continue
            
            # Step 1: Split the string into a list
        
            ingredients = row.split(',')
            
            if len(ingredients) != 35:
                ingredients = custom_split(row)
                            

            cleaned_ingredients =[
                                   convert_to_correct_type(ingredient.replace('"', '').replace('\n', '').strip())
                                   for ingredient in ingredients
                                 ]
            
            ingredients_df.loc[len(ingredients_df.index)] = cleaned_ingredients
            
            # Construct the full path to the processed data file
            processed_file_path = os.path.join(base_dir, '..', '..', 'data', 'processed_data.csv')
            
            ingredients_df.to_csv(processed_file_path, index=False)
            
        print(f"Successfully read {num_rows+1} lines from the file. Loaded {num_rows} ingredients.")
        
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

def convert_to_correct_type(value):
    try:
        # Try to convert to integer
        return int(value)
    except ValueError:
        try:
            # Try to convert to float
            return float(value)
        except ValueError:
            # If both conversions fail, return as string
            return value

# Example usage
if __name__ == "__main__":
    get_data(filename='small_data.csv')
