###############################################################################
# Prepayment model for home loan
###############################################################################

import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.0f" % x)
pd.options.display.float_format = "{:,.0f}".format


# Set up the basic variables and functions
loan_amount = 7500000
loan_term = 10
interest_rate = 9.5  # [6, 7, 8, 7, 6]  # Varying interest rates for each year
lump_sum_payment = 97048
lump_sum_interval = 1
results = []


def calculate_loan_payment(principal, interest_rate, loan_term):
    # Function to calculate the loan payment
    r = interest_rate / 100 / 12  # Monthly interest rate
    n = loan_term * 12  # Total number of months
    monthly_payment = (principal * r * (1 + r) ** n) / (((1 + r) ** n) - 1)
    return monthly_payment


###############################################################################
# Prepayment model for home loan - monthly analysis
###############################################################################


# Initialize lists to store the data
years = []
months = []
principal_list = []
interest_list = []
emi_list = []
outstanding_list = []

# Calculate the loan repayment schedule
remaining_balance = loan_amount
emi = calculate_loan_payment(remaining_balance, interest_rate, loan_term)
emi_counter = 1

for year in range(loan_term):
    for month in range(12):

        interest_payment = remaining_balance * (interest_rate / 100 / 12)
        principal_payment = emi - interest_payment
        remaining_balance -= principal_payment
        outstanding = remaining_balance

        # years.append(f"{year + 1}Y {month + 1}M")
        # months.append(f"{year + 1}Y {month + 1}M")
        years.append(f"{year + 1}")
        months.append(f"{month + 1}")
        principal_list.append(principal_payment)
        interest_list.append(interest_payment)
        emi_list.append(emi)
        outstanding_list.append(outstanding)

        if emi_counter % (lump_sum_interval * 12) == 0:
            remaining_balance -= lump_sum_payment

        emi_counter += 1

        if outstanding <= 0:
            break

    if outstanding <= 0:
        break

# Create a DataFrame to store the data
data = {
    "Year": years,
    "Month": months,
    "Principal Payment": principal_list,
    "Interest Payment": interest_list,
    "Monthly EMI": emi_list,
    "Total Outstanding": outstanding_list,
}
df_monthly = pd.DataFrame(data)
# df_monthly.iloc[-1, df_monthly.columns.get_loc('Total Outstanding ($)')] = 0

df_monthly.head()


total_interest_payment = df_monthly["Interest Payment"].sum(axis=0)
print(f"Total Interest payment = {total_interest_payment:,.2f}")

total_principal_payment = df_monthly["Principal Payment"].sum(axis=0)
print(f"Total principal payment = {total_principal_payment:,.2f}")

total_payment = total_interest_payment + total_principal_payment
print(f"Total payment = {total_payment:,.2f}")


# Plot the interest and principal components over time
# Yrs = list(range(1, (loan_term * 12) + 1))
Yrs = list(range(1, len(df_monthly) + 1))
plt.plot(Yrs, interest_list, label="Interest", color="orange")
plt.plot(Yrs, principal_list, label="Principal", color="blue")
plt.xlabel("Months")
plt.ylabel("Amount")
plt.title("Interest and Principal Components Over Time")
plt.legend()
plt.grid(True)
plt.show()


# Plot the interest and principal components over time
plt.bar(Yrs, interest_list, label="Interest", color="orange")
plt.bar(
    Yrs,
    principal_list,
    bottom=interest_list,
    label="Principal",
    color="blue",
)
plt.xlabel("Months")
plt.ylabel("Amount")
plt.title("Interest and Principal Components Over Time")
plt.legend()
plt.grid(True)
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.bar(Yrs, interest_list, label="Interest", color="orange")
ax1.bar(
    Yrs,
    principal_list,
    bottom=interest_list,
    label="Principal",
    color="blue",
)
ax1.set_xlabel("Months")
ax1.set_ylabel("Amount")
# ax1.title("Interest and Principal Components Over Time")
ax1.legend()
ax1.grid(True)

ax2.plot(Yrs, interest_list, label="Interest", color="orange")
ax2.plot(Yrs, principal_list, label="Principal", color="blue")
ax2.set_xlabel("Months")
ax2.set_ylabel("Amount")
# ax2.title("Interest and Principal Components Over Time")
ax2.legend()
ax2.grid(True)

# Adjust layout to prevent overlapping labels
plt.tight_layout()


# --------------------------------------------------------------------------

df_monthly_agg = df_monthly.groupby("Year", sort=False).sum()

Yrs1 = list(range(1, len(df_monthly_agg) + 1))
plt.plot(Yrs1, df_monthly_agg["Interest Payment"], label="Interest", color="orange")
plt.plot(Yrs1, df_monthly_agg["Principal Payment"], label="Principal", color="blue")
plt.xlabel("Years")
plt.ylabel("Amount")
plt.title("Interest and Principal Components Over Time")
plt.legend()
plt.grid(True)
plt.show()
