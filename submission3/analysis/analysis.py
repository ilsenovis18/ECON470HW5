import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
import patsy

# Load merged data
df = pd.read_csv('/Users/ilsenovis/Documents/GitHub/ECON470HW5/data/output/acs_medicaid.txt', sep='\t')

### Summarize the Data
# Q1: Calculate share of direct purchase insurance
df['share_direct'] = df['ins_direct'] / df['adult_pop']

# Group by year and calculate average share
trend = df.groupby('year', as_index=False)['share_direct'].mean()

# Plot
plt.figure(figsize=(8, 5))
plt.plot(trend['year'], trend['share_direct'], marker='o', linestyle='-', color='steelblue')
plt.title("Average Share of Adults with Direct Purchase Insurance (2012–2018)")
plt.xlabel("Year")
plt.ylabel("Share with Direct Purchase Insurance")
plt.grid(True)
plt.tight_layout()

# Save figure
plt.savefig('/Users/ilsenovis/Documents/GitHub/ECON470HW5/submission2/analysis/results_q1_direct_insurance_trend.png')

# Q3: Calculate share of adult population with Medicaid
df['share_medicaid'] = df['ins_medicaid'] / df['adult_pop']

# Group by year and calculate average share
medicaid_trend = df.groupby('year', as_index=False)['share_medicaid'].mean()

# Plot
plt.figure(figsize=(8, 5))
plt.plot(medicaid_trend['year'], medicaid_trend['share_medicaid'], marker='o', linestyle='-', color='darkgreen')
plt.title("Average Share of Adults with Medicaid Coverage (2012–2018)")
plt.xlabel("Year")
plt.ylabel("Share with Medicaid")
plt.grid(True)
plt.tight_layout()

# Save figure
plt.savefig('/Users/ilsenovis/Documents/GitHub/ECON470HW5/submission2/analysis/results_q3_medicaid_trend.png')

# Q4:

# Identify states that expanded in 2014 or never expanded
expansion_2014_states = df[df['expand_year'] == 2014]['State'].unique()
never_expanded_states = df[df['expand_ever'] == False]['State'].unique()

# Combine into one list
keep_states = list(set(expansion_2014_states).union(set(never_expanded_states)))

# Filter dataset
filtered_df = df[df['State'].isin(keep_states)].copy()

# Assign group labels
filtered_df['expansion_group'] = filtered_df['State'].apply(
    lambda x: 'Expanded in 2014' if x in expansion_2014_states else 'Did Not Expand'
)

# Calculate share of uninsured by group and year
uninsured_trend = (
    filtered_df.groupby(['year', 'expansion_group'])['uninsured']
    .sum()
    .div(filtered_df.groupby(['year', 'expansion_group'])['adult_pop'].sum())
    .reset_index(name='share_uninsured')
)

# Plot
plt.figure(figsize=(8, 5))
for label, grp in uninsured_trend.groupby('expansion_group'):
    plt.plot(grp['year'], grp['share_uninsured'], marker='o', label=label)

plt.title("Uninsured Rate Over Time by Medicaid Expansion Status")
plt.xlabel("Year")
plt.ylabel("Share Uninsured")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig('/Users/ilsenovis/Documents/GitHub/ECON470HW5/submission2/analysis/results_q4_uninsured_by_expansion.png')

### Estimate ATEs
## Q5: Difference-in-Differences Table (Uninsurance Rates)

# Filter to 2012 and 2015
df_dd = df[df['year'].isin([2012, 2015])].copy()

# Define expansion and non-expansion states
expansion_2014_states = df[df['expand_year'] == 2014]['State'].unique()
never_expanded_states = df[df['expand_ever'] == False]['State'].unique()
keep_states = list(set(expansion_2014_states).union(set(never_expanded_states)))

# Filter for relevant states
df_dd = df_dd[df_dd['State'].isin(keep_states)].copy()

# Assign group label
df_dd['group'] = df_dd['State'].apply(
    lambda x: 'Expanded' if x in expansion_2014_states else 'Not Expanded'
)

# Group and calculate average uninsurance rate
ate_table = (
    df_dd.groupby(['group', 'year'])[['uninsured', 'adult_pop']]
    .sum()
    .reset_index()
)
ate_table['uninsured_rate'] = ate_table['uninsured'] / ate_table['adult_pop']

# Pivot table for 2x2 layout
dd_pivot = ate_table.pivot(index='group', columns='year', values='uninsured_rate')
dd_pivot['Change'] = dd_pivot[2015] - dd_pivot[2012]

# Compute DiD estimate
did_estimate = dd_pivot.loc['Expanded', 'Change'] - dd_pivot.loc['Not Expanded', 'Change']

# Output
print("\nQ5: Difference-in-Differences Table (Uninsurance Rates)")
print(dd_pivot)
print(f"\nEstimated ATE (Difference-in-Differences): {did_estimate:.4f}")

# Q6 

# Restrict to 2014 expanders and never expanders only
expansion_2014_states = df[df['expand_year'] == 2014]['State'].unique()
never_expanded_states = df[df['expand_ever'] == False]['State'].unique()
keep_states = list(set(expansion_2014_states).union(set(never_expanded_states)))
df_subset = df[df['State'].isin(keep_states)].copy()

df_reg = df_subset.copy()

# Create treatment and post indicators
df_reg['treatment'] = df_reg['State'].apply(lambda x: 1 if x in expansion_2014_states else 0)
df_reg['post'] = (df_reg['year'] >= 2014).astype(int)
df_reg['interaction'] = df_reg['treatment'] * df_reg['post']
df_reg['uninsured_rate'] = df_reg['uninsured'] / df_reg['adult_pop']

# Fit OLS DiD model
model = smf.ols('uninsured_rate ~ treatment + post + interaction', data=df_reg).fit()
print(model.summary())

# Extract ATE
ate_dd = model.params['interaction']
print(f"\nEstimated ATE (DiD Regression across all years): {ate_dd:.4f}")

# Q7
df_fe = df_subset.copy()

# Create DiD variables
df_fe['treatment'] = df_fe['State'].apply(lambda x: 1 if x in expansion_2014_states else 0)
df_fe['post'] = (df_fe['year'] >= 2014).astype(int)
df_fe['interaction'] = df_fe['treatment'] * df_fe['post']
df_fe['uninsured_rate'] = df_fe['uninsured'] / df_fe['adult_pop']

# Set panel index
df_fe = df_fe.set_index(['State', 'year'])

# Run fixed effects DiD regression
model_fe = PanelOLS.from_formula(
    'uninsured_rate ~ interaction + EntityEffects + TimeEffects',
    data=df_fe
).fit()

print(model_fe.summary)
ate_fe = model_fe.params['interaction']
print(f"\nEstimated ATE (w/ FE, All Years, 2014 vs Never): {ate_fe:.4f}")

# Q8 (All States)
# Use all states and all years
df_all = df.copy()

# Fill missing expand_ever values with False (for Puerto Rico and others)
df_all['expand_ever'] = df_all['expand_ever'].fillna(False)

# Create treatment indicator: 1 if state ever expanded, else 0
df_all['treatment'] = df_all['expand_ever'].astype(int)

# Create post indicator: 1 if year >= 2014
df_all['post'] = (df_all['year'] >= 2014).astype(int)

# Interaction term
df_all['interaction'] = df_all['treatment'] * df_all['post']

# Uninsured rate outcome
df_all['uninsured_rate'] = df_all['uninsured'] / df_all['adult_pop']

# Drop rows with missing values in outcome or covariates
df_all = df_all.dropna(subset=['uninsured_rate', 'interaction'])

# Set panel index
df_all = df_all.set_index(['State', 'year'])

# Estimate DiD with fixed effects
model_q8 = PanelOLS.from_formula(
    'uninsured_rate ~ interaction + EntityEffects + TimeEffects',
    data=df_all
).fit()

# Print results
print("\nQ8: DiD Regression with All States Included (All Years, w/ FE)")
print(model_q8.summary)

# Extract ATE
ate_q8 = model_q8.params['interaction']
print(f"\nEstimated ATE (All States, w/ FE, All Years): {ate_q8:.4f}")

## Q9: Event Study – Effect of Medicaid Expansion on Uninsurance

# Subset to 2014 expanders or non-expanders only
event_df = df[df['State'].isin(keep_states)].copy()
event_df = event_df.sort_values(['State', 'year'])

# Create event time: years since expansion (0 = 2014)
event_df['event_time'] = event_df['year'] - event_df['expand_year']

# For never-expanded states, assign placeholder event_time
event_df.loc[event_df['expand_year'].isna(), 'event_time'] = -99

# Create dummies for event years -3 to 4, excluding 0 and -99
event_dummies = []
for t in range(-3, ):
    if t == 0:
        continue  # omit as reference year
    col = f"event_m{abs(t)}" if t < 0 else f"event_p{t}"
    event_df[col] = (event_df['event_time'] == t).astype(int)
    event_dummies.append(col)

# Calculate uninsurance rate
event_df['uninsured_rate'] = event_df['uninsured'] / event_df['adult_pop']

# Set panel index
event_df = event_df.set_index(['State', 'year'])

# Run panel regression with event-time dummies and fixed effects
from linearmodels.panel import PanelOLS

formula = f"uninsured_rate ~ {' + '.join(event_dummies)} + EntityEffects + TimeEffects"
event_model = PanelOLS.from_formula(formula, data=event_df, check_rank=False, drop_absorbed=True).fit()

# Extract coefficients and standard errors
coefs = event_model.params.filter(like='event_')
errors = event_model.std_errors[coefs.index]

# Convert dummy names to numeric event time
def convert_dummy_name(name):
    val = name.split('_')[1]
    return -int(val[1:]) if val.startswith('m') else int(val[1:])

years = [convert_dummy_name(name) for name in coefs.index]

# Sort estimates
sorted_pairs = sorted(zip(years, coefs, errors))
years, coefs, errors = zip(*sorted_pairs)

# Clip extreme values
coefs = pd.Series(coefs).clip(lower=-0.2, upper=0.2)
errors = pd.Series(errors).clip(upper=0.2)

# Convert event years to actual calendar years
calendar_years = [2014 + yr for yr in years]

# Plot
plt.figure(figsize=(10, 5))
plt.errorbar(calendar_years, coefs, yerr=1.96 * errors, fmt='o-', capsize=4, label='Estimated Effect')
plt.axhline(0, linestyle='--', color='gray')
plt.title("Q9: Event Study – Effect of Medicaid Expansion on Uninsurance")
plt.xlabel("Calendar Year")
plt.ylabel("Estimated Effect on Uninsured Rate")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('/Users/ilsenovis/Documents/GitHub/ECON470HW5/submission2/analysis/results_q9_event_study.png')

## Q10: All States Event Study (Aligned by Expansion Year)

# Copy and prepare data
event_df_q10 = df.copy()
event_df_q10 = event_df_q10.sort_values(['State', 'year'])

# Create event time: years since Medicaid expansion
event_df_q10['event_time'] = event_df_q10['year'] - event_df_q10['expand_year']

# Assign placeholder for never-expanded states
event_df_q10.loc[event_df_q10['expand_year'].isna(), 'event_time'] = -99

# Recalculate uninsurance rate
event_df_q10['uninsured_rate'] = event_df_q10['uninsured'] / event_df_q10['adult_pop']

# Drop NAs and infinities
event_df_q10 = event_df_q10[event_df_q10['uninsured_rate'].notna()]
event_df_q10 = event_df_q10[~event_df_q10['uninsured_rate'].isin([float('inf'), float('-inf')])]

# Create dummies for event years from -3 to +4, excluding 0 and -99
event_cols_q10 = []
for t in range(-3, 5):
    if t == 0:
        continue
    label = f'event_m{abs(t)}' if t < 0 else f'event_p{t}'
    event_df_q10[label] = (event_df_q10['event_time'] == t).astype(int)
    event_cols_q10.append(label)

# Drop the reference year (usually -1 or 0) from the regression
# In this version, we use `event_m1` as the reference, so we exclude it
event_cols_q10 = [col for col in event_cols_q10 if col != 'event_m1']

# Build regression formula
event_terms_q10 = ' + '.join(event_cols_q10)
formula_q10 = f'uninsured_rate ~ {event_terms_q10} + EntityEffects + TimeEffects'

# Set panel index
event_df_q10 = event_df_q10.set_index(['State', 'year'])

# Run regression
event_model_q10 = PanelOLS.from_formula(formula_q10, data=event_df_q10, check_rank=False, drop_absorbed=True).fit()

# Extract coefficients and errors
coefs_q10 = event_model_q10.params.filter(like='event_')
errors_q10 = event_model_q10.std_errors[coefs_q10.index]

# Convert names to event time values
def convert_event_name(name):
    if "m" in name:
        return -int(name.split('m')[1])
    elif "p" in name:
        return int(name.split('p')[1])
    return None

event_times_q10 = [convert_event_name(name) for name in coefs_q10.index]

# Sort for plotting
sorted_pairs = sorted(zip(event_times_q10, coefs_q10, errors_q10))
years_q10, coefs_q10, errors_q10 = zip(*sorted_pairs)

# Clip for readability
coefs_q10 = pd.Series(coefs_q10).clip(lower=-0.2, upper=0.2)
errors_q10 = pd.Series(errors_q10).clip(upper=0.2)

# Plot
plt.figure(figsize=(10, 5))
plt.errorbar(years_q10, coefs_q10, yerr=1.96 * errors_q10, fmt='o-', capsize=4, label='All States')
plt.axhline(0, linestyle='--', color='gray')
plt.title("Q10: Event Study – All States (Aligned by Expansion Year)")
plt.xlabel("Years Since Medicaid Expansion")
plt.ylabel("Estimated Effect on Uninsured Rate")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save
plt.savefig('/Users/ilsenovis/Documents/GitHub/ECON470HW5/submission2/analysis/results_q10_event_study_allstates.png')
# Print coefficient table
print("Q10 Event Study Coefficients:\n", coefs_q10)
