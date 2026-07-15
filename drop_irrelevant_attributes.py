#---------------------------------------------
IRRELEVANT_COLUMNS = [
    "concept:name",
    "time:timestamp",
    "case:concept:name",
    "case:Item Category",
    "User",
    "case:GR-Based Inv. Verif.",
    "case:Goods Receipt",
    "case:Document Type",
    "case:Name",
    "case:Vendor",
    "case:Source",
    "org:resource",
    "case:Purch. Doc. Category name",
    "case:Purchasing Document",
    "case:Company",
]
def drop_irrelevant_columns_hardcoded(df):
    columns_to_drop = [col for col in IRRELEVANT_COLUMNS if col in df.columns]
    return df.drop(columns=columns_to_drop)
#----------------------------------------------
def drop_irrelevant_columns_logically(df):
    """
    Drop columns that are not useful for analysis.
    This is done logically rather than by hardcoded names only.
    """
    columns_to_drop = []

    for col in df.columns:
        # Remove obvious identifiers and event metadata
        if col in {"concept:name", "time:timestamp", "case:concept:name"}:
            print(f"Remove obivious identifiers ... {col}")
            columns_to_drop.append(col)

        # Remove case-level metadata columns
        #elif col.startswith("case:") and col != "case:concept:name":
            #columns_to_drop.append(col)

        # Remove organizational/resource metadata
        elif col in {"User", "org:resource"}:
            print(f"Remove organizational/resource... {col}")
            columns_to_drop.append(col)

        # Remove constant columns
        elif df[col].nunique(dropna=False) <= 1:
            print(f"Remove constant columns... {col}")
            columns_to_drop.append(col)

    return df.drop(columns=columns_to_drop)