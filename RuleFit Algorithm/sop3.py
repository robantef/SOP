import pandas as pd
import numpy as np
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rulefit import RuleFit

warnings.filterwarnings('ignore')

from dataset import X_df, X, y, feature_names

# 2. Train RuleFit to generate a large dictionary of rules
print("Training RuleFit and extracting rule dictionary...")
rf = RuleFit(max_rules=200, max_iter=5000, tree_size=4, random_state=1)
rf.fit(X, y, feature_names=feature_names)

# 3. Extract active rules and find a nested Parent-Child pair
n_features = X.shape[1]
rule_coefs = rf.coef_[n_features:]
rules = rf.rule_ensemble.rules

# Build a dataframe of active rules
active_data = []
for rule, coef in zip(rules, rule_coefs):
    if coef != 0:
        active_data.append({'rule': str(rule), 'coef': coef})
active_rules = pd.DataFrame(active_data)
print(f"Active rules found: {len(active_rules)}")

# Helper function to parse rule strings into sets of individual conditions
def parse_conditions(rule_str):
    return set([c.strip() for c in rule_str.split('&')])

active_rules['conditions'] = active_rules['rule'].apply(parse_conditions)

parent_rule = None
child_rule = None

# Search for a strict subset relationship (Child contains all Parent conditions + more)
for i, row_a in active_rules.iterrows():
    cond_a = row_a['conditions']
    for j, row_b in active_rules.iterrows():
        cond_b = row_b['conditions']
        if cond_a.issubset(cond_b) and len(cond_a) < len(cond_b):
            parent_rule = row_a
            child_rule = row_b
            break
    if parent_rule is not None:
        break

# If no nested pair among active rules, search all rules for overlapping pair
if parent_rule is None:
    print("No strict nested pair in active rules. Searching all rules...")
    all_rules_data = []
    for rule, coef in zip(rules, rule_coefs):
        all_rules_data.append({'rule': str(rule), 'coef': coef})
    all_rules_df = pd.DataFrame(all_rules_data)
    all_rules_df['conditions'] = all_rules_df['rule'].apply(parse_conditions)
    
    for i, row_a in all_rules_df.iterrows():
        cond_a = row_a['conditions']
        for j, row_b in all_rules_df.iterrows():
            cond_b = row_b['conditions']
            if cond_a.issubset(cond_b) and len(cond_a) < len(cond_b):
                parent_rule = row_a
                child_rule = row_b
                break
        if parent_rule is not None:
            break

# Fallback print if found
if parent_rule is not None:
    print(f"Found Nested Pair!\nParent: {parent_rule['rule']} (Coef: {parent_rule['coef']:.4f})")
    print(f"Child: {child_rule['rule']} (Coef: {child_rule['coef']:.4f})")

# 4. Generate Figure 1.3: The Ceteris Paribus Violation Flowchart
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off') # Hide axes

# Format rule text for display by inserting newlines at the '&' operator
# Truncate long decimals to 5 places for readability
import re
def truncate_decimals(text, places=5):
    return re.sub(r'(\d+\.\d{' + str(places) + r'})\d+', r'\1', text)

parent_text = truncate_decimals(parent_rule['rule']).replace(' & ', '\nAND ')
child_text = truncate_decimals(child_rule['rule']).replace(' & ', '\nAND ')

# Draw Parent Rule Box
parent_box = patches.FancyBboxPatch((0.1, 0.65), 0.8, 0.2, boxstyle="round,pad=0.05", 
                                    edgecolor='darkblue', facecolor='lightblue', lw=2)
ax.add_patch(parent_box)
ax.text(0.5, 0.75, f"BROADER PARENT RULE\n\n{parent_text}\n\nLasso Weight: {parent_rule['coef']:.4f}", 
        ha='center', va='center', fontsize=11, fontweight='bold')

# Draw Child Rule Box
child_box = patches.FancyBboxPatch((0.1, 0.15), 0.8, 0.25, boxstyle="round,pad=0.05", 
                                   edgecolor='darkred', facecolor='lightcoral', lw=2)
ax.add_patch(child_box)
ax.text(0.5, 0.275, f"NESTED CHILD RULE\n\n{child_text}\n\nLasso Weight: {child_rule['coef']:.4f}", 
        ha='center', va='center', fontsize=11, fontweight='bold', color='black')

# Draw Connecting Arrow
arrow = patches.FancyArrowPatch((0.5, 0.40), (0.5, 0.65), connectionstyle="arc3,rad=0", 
                                color='black', arrowstyle='-|>', lw=2, mutation_scale=20)
ax.add_patch(arrow)

plt.title('Figure 1.3: The Ceteris Paribus Violation in Nested Rules', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figure_1_3.png', dpi=150)
plt.close()
print('Saved figure_1_3.png')
