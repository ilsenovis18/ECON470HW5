import pandas as pd

# Load cleaned ACS and Medicaid expansion datasets
acs_df = pd.read_csv('/Users/ilsenovis/Documents/GitHub/ECON470HW5/data/output/final_insurance.csv')
medicaid_df = pd.read_csv('/Users/ilsenovis/Documents/GitHub/ECON470HW5/data/output/medicaid-kff.csv')

# Merge ACS and Medicaid expansion data
acs_df = acs_df.rename(columns={'state': 'State'})
merged_df = acs_df.merge(medicaid_df, how='left', on='State')

# Create 'expand_year' and 'expand' flags
merged_df['expand_year'] = merged_df['expanded'].apply(lambda x: 2014 if x else None)
merged_df['expand'] = merged_df.apply(
    lambda row: row['year'] >= row['expand_year'] if pd.notnull(row['expand_year']) else False,
    axis=1
)

# Rename for clarity
merged_df = merged_df.rename(columns={'expanded': 'expand_ever'})

# Save merged output
merged_df.to_csv('/Users/ilsenovis/Documents/GitHub/ECON470HW5/data/output/acs_medicaid.csv', index=False)
