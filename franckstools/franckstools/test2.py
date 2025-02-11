# Import excel to panda

import pandas as pd
import numpy as np



def main():
    print("Hello World!")

    # Load the data
    path = "C:/Users/franc/Desktop/temp/Data_pour_Franck.xlsx" 

    data = pd.read_excel(path, sheet_name="Feuil1")

    data.head()

    # Remove first six column
    data = data.iloc[:, 7:]

    # Create new columns with all unique strings
    unique_strings = np.unique(list(data.values.flatten()))

    # Define a dictionary of synonyms
    synonyms = {
        'ACTranexamique': 'Acide tranexamique',
        'Acide tranexamique': 'Acide tranexamique',
        'Acide tranexanique': 'Acide tranexamique',
        'Ac.tranexamique': 'Acide tranexamique',
        'Acide tranéxamique': 'Acide tranexamique',
        'Acide tranéxamique (Cyclokapron)': 'Acide tranexamique',
        'AcideTranexamique': 'Acide tranexamique',
        'Acide_tranexamique': 'Acide tranexamique',
        'acide_tranexamique': 'Acide tranexamique',
        'Acide tranexanique ': 'Acide tranexamique',
        'Cyclokapron': 'Acide tranexamique',
        'Cyclokapron (Acide tranexamique)': 'Acide tranexamique',
        'Cyclokapron (acide tranexamique)': 'Acide tranexamique',
        'Cyklokapron': 'Acide tranexamique',
        'ac.tranexamique': 'Acide tranexamique',
        'TXA': 'Acide tranexamique',
        
        'Cyanokit': 'Cyanokit',
        'Cyanokit ': 'Cyanokit',
        'cyanokit': 'Cyanokit',
        'Hydroxycobalamine': 'Cyanokit',

        'Odansetron': 'Ondansetron',
        'Ondansetron': 'Ondansetron',
        'ondansetron': 'Ondansetron',
        'ondansétron': 'Ondansetron',
        'zofran': 'Ondansetron',

        'Phényléphrine': 'Phenylephrine',
        'phényléphrine': 'Phenylephrine',

        'Naloxone (NaloxonOrpha)': 'Naloxone',
        'naloxone': 'Naloxone',

        'Terlipressine': 'Terlipressin',
        'terlipressine': 'Terlipressin',



    }

    # Replace synonyms in each column
    for column in data.columns:
        data[column] = data[column].map(synonyms).fillna(data[column])
    # Fill the new dataframe with the number of occurences of each medicament for each patient
    
    # Replace old unique strings with new unique strings
    unique_strings = np.unique(list(data.values.flatten()))

    # Create new dataframe with patient ID for columns and medicament name for rows
    new_data = pd.DataFrame(columns=data.columns, index=unique_strings)
    
    for patient in data.columns:
        for medicament in unique_strings:
            new_data.loc[medicament, patient] = list(data[patient].values).count(medicament)

    
    print("Hello")

    new_data[14478476]['Adrénaline']

    # Create new excel file with patient Id for columns and medicament name for columns
    # switch rows and columns
    new_data = new_data.T
    new_data.to_excel("C:/Users/franc/Desktop/temp/Data_pour_Franck_new.xlsx")


if __name__ == "__main__":
    main()