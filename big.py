import streamlit as st
import pandas as pd
from apyori import apriori

# Data Preparation
data = {
    'Transaction': [1, 2, 3, 4, 5],
    'Items': [['Milk', 'Bread'],
              ['Butter', 'Bread'],
              [ 'Bread', 'Butter'],
              ['Milk', 'Bread', 'Butter'],
              ['Milk', 'Bread', 'Butter']]
}
df = pd.DataFrame(data)

# Display Transactions
st.title("Transaction Data")
st.write("Below are the transactions used for Association Rule Mining:")
st.dataframe(df)

# Convert transactions to a list of lists for Apriori
transactions = df['Items'].tolist()

# Apply Apriori Algorithm
rules = apriori(
    transactions=transactions,
    min_support=0.1,
    min_confidence=0.2,
    min_lift=1.0,
    min_length=2
)
results = list(rules)

# Inspect and extract rule details
def inspect(results):
    product1 = []
    product2 = []
    supports = []
    confidences = []
    lifts = []
    for result in results:
        for stat in result.ordered_statistics:
            if len(stat.items_base) == 1 and len(stat.items_add) == 1:
                product1.append(list(stat.items_base)[0])
                product2.append(list(stat.items_add)[0])
                supports.append(result.support)
                confidences.append(stat.confidence)
                lifts.append(stat.lift)
    return list(zip(product1, product2, supports, confidences, lifts))

# Generate DataFrame for Rules
if results:
    st.write("Rules generated:")
    df_results = pd.DataFrame(
        inspect(results),
        columns=['Product 1', 'Product 2', 'Support', 'Confidence', 'Lift']
    )

    # Convert Lift column to numeric
    df_results['Lift'] = pd.to_numeric(df_results['Lift'], errors='coerce')
    df_results = df_results.dropna(subset=['Lift'])

    # Sort by Lift
    df_sorted = df_results.nlargest(n=10, columns='Lift')

    # Display Rules
    st.title("Association Rule Mining Results")
    st.write("Top Rules Based on Lift:")
    st.dataframe(df_sorted)
else:
    st.title("No Rules Generated")
    st.write("Try adjusting the Apriori parameters to generate rules.")
