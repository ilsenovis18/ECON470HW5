import os
import pandas as pd
from census import Census
from us.states import STATES

# Get key securely from the environment
api_key = os.getenv("APIKEY")
c = Census(api_key)

# Define the ACS variable codes
variables = [
    "B27010_018E", "B27010_020E", "B27010_021E", "B27010_022E",
    "B27010_023E", "B27010_024E", "B27010_025E", "B27010_033E",
    "B27010_034E", "B27010_036E", "B27010_037E", "B27010_038E",
    "B27010_039E", "B27010_040E", "B27010_041E", "B27010_050E"
]

# Mapping from ACS variable codes to readable names
rename_map = {
    "B27010_018E": "all_18to34",
    "B27010_020E": "employer_18to34",
    "B27010_021E": "direct_18to34",
    "B27010_022E": "medicare_18to34",
    "B27010_023E": "medicaid_18to34",
    "B27010_024E": "tricare_18to34",
    "B27010_025E": "va_18to34",
    "B27010_033E": "none_18to34",
    "B27010_034E": "all_35to64",
    "B27010_036E": "employer_35to64",
    "B27010_037E": "direct_35to64",
    "B27010_038E": "medicare_35to64",
    "B27010_039E": "medicaid_35to64",
    "B27010_040E": "tricare_35to64",
    "B27010_041E": "va_35to64",
    "B27010_050E": "none_35to64"
}

# Download and combine data for each year
all_years_data = []

for year in range(2012, 2019):
    print(f"Fetching data for year {year}...")
    for state in STATES:
        print(f"  - State: {state.name}")
        state_data = c.acs1.state(variables, state.fips, year=year)
        for row in state_data:
            row["state"] = state.name
            row["year"] = year
        all_years_data.extend(state_data)

df = pd.DataFrame(all_years_data)
df = df.rename(columns=rename_map)

# Calculate new summary variables
df["adult_pop"] = df["all_18to34"] + df["all_35to64"]
df["ins_employer"] = df["employer_18to34"] + df["employer_35to64"]
df["ins_direct"] = df["direct_18to34"] + df["direct_35to64"]
df["ins_medicare"] = df["medicare_18to34"] + df["medicare_35to64"]
df["ins_medicaid"] = df["medicaid_18to34"] + df["medicaid_35to64"]
df["uninsured"] = df["none_18to34"] + df["none_35to64"]

# Keep only the relevant columns
df = df[["state", "year", "adult_pop", "ins_employer", "ins_direct",
         "ins_medicare", "ins_medicaid", "uninsured"]]

# Save the final result
output_path = "/Users/ilsenovis/Documents/GitHub/ECON470HW5/data/output/final_insurance.csv"
df.to_csv(output_path, index=False)
