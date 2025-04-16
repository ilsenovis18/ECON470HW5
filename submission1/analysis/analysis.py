import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
import patsy

# Load merged data
df = pd.read_csv('/Users/ilsenovis/Documents/GitHub/ECON470HW5/data/output/acs_medicaid.csv')

### Summarize the Data
## Q1
# Calculate share of direct purchase insurance
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
plt.savefig('/Users/ilsenovis/Documents/GitHub/ECON470HW5/submission1/analysis/results_q1_direct_insurance_trend.png')

## Q3
# Calculate share of adult population with Medicaid
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
plt.savefig('/Users/ilsenovis/Documents/GitHub/ECON470HW5/submission1/analysis/results_q3_medicaid_trend.png')

## Q4
# Identify states that expanded in 2014 or never expanded
expansion_2014_states = df[df['expand_year'] == 2014]['State'].unique()
never_expanded_states = df[df['expand_ever'] == False]['State'].unique()

# Combine them into one list
keep_states = list(set(expansion_2014_states).union(set(never_expanded_states)))

# Filter dataset
filtered_df = df[df['State'].isin(keep_states)].copy()

# Assign group labels explicitly
filtered_df['expansion_group'] = filtered_df['State'].apply(
    lambda x: 'Expanded in 2014' if x in expansion_2014_states else 'Did Not Expand'
)

# Group by year and expansion status, then average
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
plt.savefig('/Users/ilsenovis/Documents/GitHub/ECON470HW5/submission1/analysis/results_q4_uninsured_by_expansion.png')

### Estimate ATEs
## Q5
# Filter to 2012 and 2015
df_dd = df[df['year'].isin([2012, 2015])].copy()

# Keep only states that expanded in 2014 or never expanded
expansion_2014_states = df[df['expand_year'] == 2014]['State'].unique()
never_expanded_states = df[df['expand_ever'] == False]['State'].unique()
keep_states = list(set(expansion_2014_states).union(set(never_expanded_states)))
df_dd = df_dd[df_dd['State'].isin(keep_states)].copy()

# Assign treatment group
df_dd['group'] = df_dd['State'].apply(
    lambda x: 'Expanded' if x in expansion_2014_states else 'Not Expanded'
)

# Calculate average uninsurance rate by year and group
ate_table = (
    df_dd.groupby(['group', 'year'])[['uninsured', 'adult_pop']]
    .sum()
    .reset_index()
)

ate_table['uninsured_rate'] = ate_table['uninsured'] / ate_table['adult_pop']

# Pivot for 2x2 display
dd_pivot = ate_table.pivot(index='group', columns='year', values='uninsured_rate')
dd_pivot['Change'] = dd_pivot[2015] - dd_pivot[2012]

# DiD estimate
did_estimate = dd_pivot.loc['Expanded', 'Change'] - dd_pivot.loc['Not Expanded', 'Change']

# Display
print("\nQ5: Difference-in-Differences Table (Uninsurance Rates)")
print(dd_pivot)
print(f"\nEstimated ATE (Difference-in-Differences): {did_estimate:.4f}")

## Q6
# Use all years
df_reg = df[df['year'].isin([2012, 2015])].copy()
df_reg = df_reg[df_reg['State'].isin(keep_states)].copy()

# Create DiD variables
df_reg['treatment'] = df_reg['State'].apply(lambda x: 1 if x in expansion_2014_states else 0)
df_reg['post'] = df_reg['year'].apply(lambda x: 1 if x == 2015 else 0)
df_reg['interaction'] = df_reg['treatment'] * df_reg['post']

# Outcome: uninsurance rate
df_reg['uninsured_rate'] = df_reg['uninsured'] / df_reg['adult_pop']

# Fit the regression: uninsurance ~ treatment + post + interaction
model = smf.ols('uninsured_rate ~ treatment + post + interaction', data=df_reg).fit()

# Display results
print("\nQ6: DiD Regression Results")
print(model.summary())

# Extract ATE from the interaction term
ate_dd = model.params['interaction']
print(f"\nEstimated ATE (DiD Regression): {ate_dd:.4f}")

## Q7
# Prepare panel data
df_panel = df[df['year'].isin([2012, 2015])].copy()
df_panel = df_panel[df_panel['State'].isin(keep_states)].copy()

# Set panel index (state-year)
df_panel = df_panel.set_index(['State', 'year'])

# Create treatment variables
df_panel['treatment'] = df_panel['expand_ever'].astype(int)
df_panel['post'] = (df_panel.index.get_level_values('year') == 2015).astype(int)
df_panel['interaction'] = df_panel['treatment'] * df_panel['post']
df_panel['uninsured_rate'] = df_panel['uninsured'] / df_panel['adult_pop']

# Estimate with state and year fixed effects
model_fe = PanelOLS.from_formula(
    formula="uninsured_rate ~ interaction + EntityEffects + TimeEffects",
    data=df_panel
).fit()

print("\nQ7: DiD Regression with State and Year Fixed Effects")
print(model_fe.summary)

# Extract ATE
ate_fe = model_fe.params['interaction']
print(f"\nEstimated ATE (w/ FE): {ate_fe:.4f}")

## Q8
# Create a fresh copy using ALL states (not filtered to keep_states)
df_all = df[df['year'].isin([2012, 2015])].copy()

# Create treatment indicator (1 if expanded by 2015, 0 otherwise)
df_all['treatment'] = df_all['expand_ever'].astype(int)
df_all['post'] = (df_all['year'] == 2015).astype(int)
df_all['interaction'] = df_all['treatment'] * df_all['post']
df_all['uninsured_rate'] = df_all['uninsured'] / df_all['adult_pop']

# Set index for panel model
df_all = df_all.set_index(['State', 'year'])

# Run fixed effects regression
from linearmodels.panel import PanelOLS

model_all = PanelOLS.from_formula(
    'uninsured_rate ~ interaction + EntityEffects + TimeEffects',
    data=df_all
).fit()

print("\nQ8: DiD Regression with All States Included (w/ FE)")
print(model_all.summary)

# Extract ATE
ate_all = model_all.params['interaction']
print(f"\nEstimated ATE (All States, w/ FE): {ate_all:.4f}")

## Q9
# Subset data to states that expanded in 2014 or never expanded
event_df = df[df['State'].isin(keep_states)].copy()
event_df = event_df.sort_values(['State', 'year'])

# Create event time: years since expansion (0 = 2014)
event_df['event_time'] = event_df['year'] - event_df['expand_year']

# For never-expanded states, assign a fake constant event_time
event_df.loc[event_df['expand_year'].isna(), 'event_time'] = -99

# Create dummy variables for event years -3 to 4 (excluding 0 and -99),
# but rename them without minus signs
for t in range(-3, 5):
    if t == 0:
        continue  # skip reference year
    # Create a new name: if t is negative, use "m" for minus; if positive, "p".
    if t < 0:
        newname = f'event_m{abs(t)}'
    else:
        newname = f'event_p{t}'
    event_df[newname] = (event_df['event_time'] == t).astype(int)

# Build list of new dummy variable names (only include those that were actually created)
event_cols = [col for col in event_df.columns if col.startswith('event_') and col not in ['event_time', 'event_m1']]
print("Final event columns used in formula:", event_cols)

expanders = event_df[event_df['expand_year'] == 2014]
expanders = expanders.copy()
expanders['uninsured_rate'] = expanders['uninsured'] / expanders['adult_pop']
expanders = expanders[expanders['uninsured_rate'].notna() & ~expanders['uninsured_rate'].isin([float('inf'), float('-inf')])]
print(expanders[['uninsured', 'adult_pop', 'uninsured_rate']].describe())
non_expanders = event_df[event_df['expand_year'].isna()]

# Estimate separate regressions for expanders and non-expanders
# Recalculate uninsurance rate
expanders = expanders.copy()
expanders['uninsured_rate'] = expanders['uninsured'] / expanders['adult_pop']
expanders = expanders[expanders['uninsured_rate'].notna() & ~expanders['uninsured_rate'].isin([float('inf'), float('-inf')])]
print(expanders[['uninsured', 'adult_pop', 'uninsured_rate']].describe())

# Set index
expanders = expanders.set_index(['State', 'year'])
# non_expanders = non_expanders.set_index(['State', 'year'])  # Removed

# Add this line before # Build formulas
event_terms = ' + '.join(event_cols)

# Build formulas
expand_formula = f'uninsured_rate ~ {event_terms} + EntityEffects + TimeEffects'
# non_expand_formula = f'uninsured_rate ~ {event_terms} + EntityEffects + TimeEffects'  # Removed

# Run regressions
expand_model = PanelOLS.from_formula(expand_formula, data=expanders, check_rank=False, drop_absorbed=True).fit()
# non_expand_model = PanelOLS.from_formula(non_expand_formula, data=non_expanders, check_rank=False, drop_absorbed=True).fit()  # Removed

# Extract coefficients and errors
expand_coefs = expand_model.params.filter(like='event_')
expand_errors = expand_model.std_errors[expand_coefs.index]


# Convert names to years
def convert_name(name):
    if "m" in name:
        return -int(name.split('m')[1])
    elif "p" in name:
        return int(name.split('p')[1])
    return None

expand_years = [convert_name(name) for name in expand_coefs.index]

# Only build and run model if dummies exist
if event_cols:
    event_terms = ' + '.join(event_cols)
    formula = f'uninsured_rate ~ {event_terms} + EntityEffects + TimeEffects'
    
    # Recalculate uninsurance rate
    event_df['uninsured_rate'] = event_df['uninsured'] / event_df['adult_pop']
    
    # Set panel index for the regression
    event_df = event_df.set_index(['State', 'year'])
    
    # Debug: Print formula and dataframe columns
    print("EVENT FORMULA:", formula)
    print("COLUMNS IN DATAFRAME:", event_df.columns.tolist())
    
    # Run the fixed effects regression using PanelOLS
    event_model = PanelOLS.from_formula(formula, data=event_df, check_rank=False, drop_absorbed=True).fit()
    
    # Extract coefficients and standard errors for our event dummies
    coefs = event_model.params.filter(like='event_')
    errors = event_model.std_errors[coefs.index]
    # Convert dummy names back to event time for plotting:
    # e.g., event_m3 becomes -3; event_p1 becomes 1.
    def convert_dummy_name(name):
        suffix = name.split('_')[1]
        if suffix.startswith('m'):
            return -int(suffix[1:])
        elif suffix.startswith('p'):
            return int(suffix[1:])
        else:
            return None
    years = [convert_dummy_name(name) for name in coefs.index]
    
    # Filter out outlier effects if necessary
coefs = coefs.clip(lower=-0.2, upper=0.2)
errors = errors.clip(upper=0.2)            

# Recompute year labels
years = [2014 + e for e in expand_years]

plt.figure(figsize=(10, 5))
plt.errorbar(years, coefs, yerr=1.96 * errors, fmt='o-', capsize=4, label='Expanded in 2014 (Event Study)')
plt.axhline(0, linestyle='--', color='gray')
plt.title("Q9: Event Study – Effect of Medicaid Expansion on Uninsurance")
plt.xlabel("Calendar Year")
plt.ylabel("Estimated Effect on Uninsured Rate")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save or show plot
plt.savefig('/Users/ilsenovis/Documents/GitHub/ECON470HW5/submission1/analysis/results_q9_event_study.png')

# Optional: print diagnostics
print("Event Study Coefficients:\n", coefs)
print("Standard Errors:\n", errors)

## Q10
# Include all states (including those that expanded after 2014)
event_df_q10 = df.copy()
event_df_q10 = event_df_q10.sort_values(['State', 'year'])

# Create event time
event_df_q10['event_time'] = event_df_q10['year'] - event_df_q10['expand_year']
event_df_q10.loc[event_df_q10['expand_year'].isna(), 'event_time'] = -99  # constant for never-expanded

# Create dummy variables for event years -3 to +4 (excluding 0 and -99), renamed appropriately
for t in range(-3, 5):
    if t == 0:
        continue
    label = f'event_m{abs(t)}' if t < 0 else f'event_p{t}'
    event_df_q10[label] = (event_df_q10['event_time'] == t).astype(int)

# Build list of event columns
event_cols_q10 = [col for col in event_df_q10.columns if col.startswith('event_') and col not in ['event_time', 'event_m1']]
print("Q10 Event columns used in formula:", event_cols_q10)

# Prepare formula and set index
event_terms_q10 = ' + '.join(event_cols_q10)
formula_q10 = f'uninsured_rate ~ {event_terms_q10} + EntityEffects + TimeEffects'

# Calculate uninsurance rate
event_df_q10['uninsured_rate'] = event_df_q10['uninsured'] / event_df_q10['adult_pop']
event_df_q10 = event_df_q10[event_df_q10['uninsured_rate'].notna()]
event_df_q10 = event_df_q10.set_index(['State', 'year'])

# Fit model
event_model_q10 = PanelOLS.from_formula(formula_q10, data=event_df_q10, check_rank=False, drop_absorbed=True).fit()

# Extract coefficients and standard errors
coefs_q10 = event_model_q10.params.filter(like='event_')
errors_q10 = event_model_q10.std_errors[coefs_q10.index]

# Convert names to calendar years relative to expansion
def convert_q10(name):
    if "m" in name:
        return 2014 - int(name.split('m')[1])
    elif "p" in name:
        return 2014 + int(name.split('p')[1])
    return None

years_q10 = [convert_q10(name) for name in coefs_q10.index]

# Clip for clarity
coefs_q10 = coefs_q10.clip(lower=-0.2, upper=0.2)
errors_q10 = errors_q10.clip(upper=0.2)

# Plot
plt.figure(figsize=(10, 5))
plt.errorbar(years_q10, coefs_q10, yerr=1.96 * errors_q10, fmt='o-', capsize=4, label='All States (Event Study)')
plt.axhline(0, linestyle='--', color='gray')
plt.title("Q10: Event Study – All States Aligned on Medicaid Expansion")
plt.xlabel("Calendar Year")
plt.ylabel("Estimated Effect on Uninsured Rate")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save
plt.savefig('/Users/ilsenovis/Documents/GitHub/ECON470HW5/submission1/analysis/results_q10_event_study_allstates.png')
print("Q10 Event Study Coefficients:\n", coefs_q10)
