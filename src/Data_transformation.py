import pandas as pd

# Run from the repo folder!!

# Import the data
df = pd.read_excel("data/AML Assignment data_std.xlsx", sheet_name = "data")

# Drop the ID: every customer is unique, so it is not needed for prediction
df_mod = df.drop(columns = "cust_id_masked")

# Binary indicators were standardized, so put them back to 0 - 1, then categorical
binary_cols = df_mod.columns[df_mod.nunique() == 2]

for col in binary_cols:
    df_mod[col] = (df_mod[col] == df_mod[col].max()).astype(int)

    df_mod[col] = df_mod[col].astype("category")

# tot_acct_num and tot_acct_num.1 are exact copies of prod_prim_acct_cnt
# geo_domicile_country_rate_* are exact copies of geo_mail_addr_country_rate_* in this dataset
df_mod = df_mod.drop(columns = ["tot_acct_num", "tot_acct_num.1",
                                "geo_domicile_country_rate_high",
                                "geo_domicile_country_rate_med",
                                "geo_domicile_country_rate_low"])

# Counts were standardized too, so put them back to whole numbers (smallest gap = 1, smallest value = 0)
# Checked with the totals, ex: prod_total_acct_cnt = prod_prim_acct_cnt + prod_none_prim_acct_cnt
count_cols = [col for col in df_mod.columns if "_cnt" in col or col.endswith("_num")]

for col in count_cols:
    gap = df_mod[col].drop_duplicates().sort_values().diff().min()
    df_mod[col] = ((df_mod[col] - df_mod[col].min()) / gap).round().astype(int)

# Export as pickle because, unlike a csv, it keeps the category type
df_mod.to_pickle("data/AML_data_transformed.pkl")
