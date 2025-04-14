import pandas as pd
from datetime import datetime

# Load the data
kff_dat = pd.read_csv(
    '/Users/ilsenovis/Documents/GitHub/ECON470HW5/data/input/raw_data.csv', 
    skiprows=2,
    nrows=53)

# Rename columns for clarity
kff_dat.columns = ['State', 'Expansion Status']

# Create binary flag for Medicaid expansion
kff_dat['expanded'] = kff_dat['Expansion Status'] == 'Adopted'

# Save cleaned file
kff_dat[['State', 'expanded']].to_csv(
    '/Users/ilsenovis/Documents/GitHub/ECON470HW5/data/output/medicaid-kff.csv',
    index=False
)