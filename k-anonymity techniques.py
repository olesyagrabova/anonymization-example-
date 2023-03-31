import pandas as pd
import numpy as np

np.random.seed(42)

# Generate synthetic data with more controlled distributions (1,000 rows)
age = np.random.choice([25, 30, 35, 40, 45, 50, 55, 60, 65, 70], size=1000, p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
sex = np.random.choice(['M', 'F'], size=1000, p=[0.5, 0.5])
blood_type = np.random.choice(['A', 'B', 'AB', 'O'], size=1000, p=[0.4, 0.3, 0.2, 0.1])
smoker = np.random.choice([0, 1], size=1000, p=[0.7, 0.3])
height = np.random.choice([150, 160, 170, 180, 190], size=1000, p=[0.2, 0.3, 0.3, 0.1, 0.1])
weight = np.random.choice([60, 70, 80, 90, 100], size=1000, p=[0.2, 0.3, 0.3, 0.1, 0.1])
phone_numbers = ['555-555-' + '{0:04d}'.format(x) for x in range(1000)]

# Create a Pandas dataframe
data = pd.DataFrame({'Age': age, 'Sex': sex, 'Blood_Type': blood_type, 'Smoker': smoker, 'Height': height, 'Weight': weight, 'Phone_Number': phone_numbers})
print(data)

df_backup = data.copy()
print(df_backup)

def queryKAnonymized(row):
    return f'Age == {row.Age}' \
           f' & Sex == \'{row.Sex}\'' \
           f' & Blood_Type == \'{row.Blood_Type}\'' \
           f' & Smoker == {row.Smoker}' \
           f' & Height == {row.Height}' \
           f' & Weight == {row.Weight}'\
           f' & Phone_Number == \'{row.Phone_Number}\''

def isKAnonymized(df, k, queryFunction = queryKAnonymized):
    for index, row in df.iterrows():
        if df.query(queryFunction(row)).shape[0] < k:
            return False
    return True

print(isKAnonymized(data, 1))
print(isKAnonymized(data, 2))

def getNotKAnonymized(df, k, queryFunction = queryKAnonymized):
    rowsNotKAnonymized = pd.DataFrame()
    for index, row in df.iterrows():
        group = df.query(queryFunction(row))
        if group.shape[0] < k: 
            rowsNotKAnonymized = pd.concat([rowsNotKAnonymized, group])
    return rowsNotKAnonymized.drop_duplicates()

print(getNotKAnonymized(data, 1))
print(getNotKAnonymized(data, 2))

# Mask phone number
data['Phone_Number'] = data['Phone_Number'].apply(lambda x: x[:-8] + '*'*8)

def queryKAnonymized_masked(row):
    return f'Age_Range == \'{row.Age_Range}\'' \
           f' & Sex == \'{row.Sex}\'' \
           f' & Blood_Type == \'{row.Blood_Type}\'' \
           f' & Smoker == {row.Smoker}' \
           f' & Height_Range == \'{row.Height_Range}\'' \
           f' & Weight_Range == \'{row.Weight_Range}\''\
           f' & Phone_Number == \'{row.Phone_Number}\''

# Use the anonymized variables in the anonymized dataset
data_masked = data[['Age_Range', 'Sex', 'Blood_Type', 'Smoker', 'Height_Range', 'Weight_Range', 'Phone_Number']]

print(data_masked)

len(getNotKAnonymized(data_masked,2,queryKAnonymized_masked))
#27

# Generalize age into broader age ranges
data['Age_Range'] = pd.cut(data['Age'], bins=[18, 35, 65, 100], labels=['18-35', '36-65', '66-100'])

# Generalize height into broader height ranges
data['Height_Range'] = pd.cut(data['Height'], bins=[140, 170, 210], labels=['140-170', '171-210'])

# Generalize weight into broader weight ranges
data['Weight_Range'] = pd.cut(data['Weight'], bins=[40, 70, 100], labels=['40-70', '71-100'])


def queryKAnonymized_masked_generalised(row):
    return f'Age_Range == \'{row.Age_Range}\'' \
           f' & Sex == \'{row.Sex}\'' \
           f' & Blood_Type == \'{row.Blood_Type}\'' \
           f' & Smoker == {row.Smoker}' \
           f' & Height_Range == \'{row.Height_Range}\'' \
           f' & Weight_Range == \'{row.Weight_Range}\''\
           f' & Phone_Number == \'{row.Phone_Number}\''

# Use the anonymized variables in the anonymized dataset
data_masked_generalised = data[['Age_Range', 'Sex', 'Blood_Type', 'Smoker', 'Height_Range', 'Weight_Range', 'Phone_Number']]
print(data_masked_generalised)

len(getNotKAnonymized(data_masked_generalised,2,queryKAnonymized_masked_generalised))

#27


# Drop Smoker column
data_suppressed = data.drop(columns=['Smoker'])

# Generalize variables in the anonymized dataset
data_masked_generalised_supressed = data_suppressed[['Age_Range', 'Sex','Blood_Type','Height_Range', 'Weight_Range','Phone_Number']]
print(data_masked_generalised_supressed)

def queryKAnonymized_masked_generalised_supressed(row):
    return f'Age_Range == \'{row.Age_Range}\'' \
           f' & Sex == \'{row.Sex}\'' \
           f' & Blood_Type == \'{row.Blood_Type}\'' \
           f' & Height_Range == \'{row.Height_Range}\'' \
           f' & Weight_Range == \'{row.Weight_Range}\''\
           f' & Phone_Number == \'{row.Phone_Number}\''

len(getNotKAnonymized(data_masked_generalised_supressed,2,queryKAnonymized_masked_generalised_supressed))

# Drop Phone Number and Blood Type columns
data_suppressed = data.drop(columns=['Phone_Number', 'Blood_Type'])

# Generalize variables in the anonymized dataset
data_masked_generalised_supressed = data_suppressed[['Age_Range', 'Sex', 'Height_Range', 'Weight_Range']]
print(data_masked_generalised_supressed)

def queryKAnonymized_masked_generalised_supressed(row):
    return f'Age_Range == \'{row.Age_Range}\'' \
           f' & Sex == \'{row.Sex}\'' \
           f' & Height_Range == \'{row.Height_Range}\'' \
           f' & Weight_Range == \'{row.Weight_Range}\''

len(getNotKAnonymized(data_masked_generalised_supressed,2,queryKAnonymized_masked_generalised_supressed))
#0

# Check k-anonymity and number of not anonymized rows
k_values = [1, 2, 3, 4, 5]

not_k_anonymized_rows = []

for k in k_values:
    anonymized = isKAnonymized(data_masked_generalised_supressed, k, queryKAnonymized_masked_generalised_supressed)
    not_k_anonymized_rows.append(len(getNotKAnonymized(data_masked_generalised_supressed, k, queryKAnonymized_masked_generalised_supressed)))
    print(f"For k = {k}, is the dataset k-anonymized? {anonymized}. Number of not k-anonymized rows: {not_k_anonymized_rows[-1]}")

print(f"Number of not k-anonymized rows for k={k_values}: {not_k_anonymized_rows}")

#For k = 1, is the dataset k-anonymized? True. Number of not k-anonymized rows: 0
#For k = 2, is the dataset k-anonymized? True. Number of not k-anonymized rows: 0
#For k = 3, is the dataset k-anonymized? True. Number of not k-anonymized rows: 0
#For k = 4, is the dataset k-anonymized? False. Number of not k-anonymized rows: 1
#For k = 5, is the dataset k-anonymized? False. Number of not k-anonymized rows: 2

# Drop the not k-anonymized rows
not_k_anonymized_rows_4 = getNotKAnonymized(data_masked_generalised_supressed, 4, queryKAnonymized_masked_generalised_supressed).index
not_k_anonymized_rows_5 = getNotKAnonymized(data_masked_generalised_supressed, 5, queryKAnonymized_masked_generalised_supressed).index

# Drop the not k-anonymized rows
data_k_anonymized = data_masked_generalised_supressed.drop(np.concatenate([not_k_anonymized_rows_4, not_k_anonymized_rows_5]))

# Iterate until all non-k-anonymized rows are dropped
while True:
    not_k_anonymized_rows = []
    for k in k_values:
        not_k_anonymized_rows.append(getNotKAnonymized(data_k_anonymized, k, queryKAnonymized_masked_generalised_supressed).index)
    not_k_anonymized_rows = np.concatenate(not_k_anonymized_rows)
    if len(not_k_anonymized_rows) == 0:
        break
    data_k_anonymized = data_k_anonymized.drop(not_k_anonymized_rows)

# Check k-anonymity and number of not anonymized rows
k_values = [1, 2, 3, 4, 5]

not_k_anonymized_rows = []

for k in k_values:
    anonymized = isKAnonymized(data_k_anonymized, k, queryKAnonymized_masked_generalised_supressed)
    not_k_anonymized_rows.append(len(getNotKAnonymized(data_k_anonymized, k, queryKAnonymized_masked_generalised_supressed)))
    print(f"For k = {k}, is the dataset k-anonymized? {anonymized}. Number of not k-anonymized rows: {not_k_anonymized_rows[-1]}")

print(f"Number of not k-anonymized rows for k={k_values}: {not_k_anonymized_rows}")

print(data_k_anonymized)




