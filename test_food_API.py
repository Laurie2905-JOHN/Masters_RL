import csv
import requests
from xml.etree import ElementTree

# Function to get and parse data from the given URL
def get_and_parse_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        # Parse the XML response
        root = ElementTree.fromstring(response.content)
        return root
    else:
        print(f"Error: Received status code {response.status_code}")
        return None

# Function to extract data from XML and write to CSV, including nutritional values
def extract_data_and_write_csv(food_root, nutritional_root, csv_file_path):
    # Open a CSV file for writing
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['Number', 'Name', 'WeightGrams', 'ClassificationName', 'Code', 'Nutrient', 'Amount']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        # Iterate through each "Livsmedel" element in the food classification XML
        for livsmedel in food_root.findall('.//Livsmedel'):
            number = livsmedel.find('Nummer').text
            name = livsmedel.find('Namn').text  # Placeholder for translation
            weight_grams = livsmedel.find('ViktGram').text
            
            # Iterate through each "Klassificering" within "Livsmedel"
            for klassificering in livsmedel.findall('.//Klassificering'):
                
                
                classification_name = klassificering.find('Namn').text  # Placeholder for translation
                # There might be multiple "KlassificeringVarde" elements
                for varde in klassificering.findall('.//KlassificeringVarde'):
                    code = varde.find('Kod').text
                    
                    # Find corresponding nutritional values for this food item
                    for nutrient in nutritional_root.findall(f".//Livsmedel[Nummer='{number}']/Naringsvarde"):
                        nutrient_name = nutrient.find('Namn').text  # Placeholder for translation
                        amount = nutrient.find('Mangd').text
                        
                        # Write row to CSV, including nutrient and amount
                        writer.writerow({
                            'Number': number, 'Name': name, 'WeightGrams': weight_grams,
                            'ClassificationName': classification_name, 'Code': code,
                            'Nutrient': nutrient_name, 'Amount': amount
                        })
    print("CSV file has been created successfully with nutritional values.")

# Example usage
date = "20170101"
url_classification = f"http://www7.slv.se/apilivsmedel/LivsmedelService.svc/Livsmedel/Klassificering/{date}"
url_nutritional_values = f"http://www7.slv.se/apilivsmedel/LivsmedelService.svc/Livsmedel/Naringsvarde/{date}"

food_root = get_and_parse_data(url_classification)
nutritional_root = get_and_parse_data(url_nutritional_values)

if food_root is not None and nutritional_root is not None:
    csv_file_path = 'full_food_data.csv'
    extract_data_and_write_csv(food_root, nutritional_root, csv_file_path)
