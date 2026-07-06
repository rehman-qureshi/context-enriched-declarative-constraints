import pandas as pd
from find_shapley_values import conditional_shapley_with_binning
from build_decision_trees import build_decison_trees_function
#------------------------------
def has_alternate_precedence(events, A_list, B_list):
    """Return True if the trace satisfies AlternatePrecedence(A,B).

    Alternate Precedence means:
    - Every occurrence of an event in B_list must be preceded by at least an event in A_list.
    - Between two consecutive B events there must be at least one A event.

    This implementation treats traces with no B events as not satisfying the constraint,
    because this script classifies cases with first event but no second event as False.
    """
    last_relevant = None
    has_b = False

    for e in events:
        if e in A_list:
            last_relevant = "A"
        elif e in B_list:
            has_b = True
            # B must be preceded by an A, and the last relevant event must be A
            if last_relevant != "A":
                return False
            last_relevant = "B"

    return has_b
#------------------------------
def has_at_most_one(events, A_list):
    """Return True if the trace satisfies AtMostOne(A).

    At Most One means:
    - For every event type listed in A_list, there can be at most one occurrence in the trace.
    """
    for a in A_list:
        if events.count(a) > 1:
            return False
    return True
# -----------------------------
def has_end(events, A_list):
    """Return True if the trace satisfies End(A).

    End means:
    - The last event in the trace must be an event in A_list.
    """
    return events[-1] in A_list if events else False
#--------------------------------------------
def check_conformance_function(event_logs, constraint):
    
    print(f"Calculating Shapley value for constraint: {constraint['id']} - {constraint['template']}")
    shap_values = {}
    if constraint.get("type") == "Binary":
        print(f"Source activities: {constraint['source']}")
        print(f"Target activities: {constraint['target']}")
        first_events = constraint['source']
        second_events = constraint['target']
        # Create a copy of the event logs DataFrame to avoid modifying the original
        df=event_logs.copy()
        grouped = df.groupby("case:concept:name")["concept:name"].agg(list)
        # Cases that contain the second event at least once (i.e. B occurs)
        cases_with_second_event = grouped[grouped.apply(lambda x: any(e in x for e in second_events))].index

        # Cases that satisfy AlternatePrecedence(A,B)
        cases_satisfying_alternate_precedence = grouped[grouped.apply(
        lambda x: has_alternate_precedence(x, A_list=first_events, B_list=second_events)    
        )].index

        print("Total cases:", len(grouped))
        print("Cases with B (second event):", len(cases_with_second_event))
        print("Cases satisfying AlternatePrecedence:", len(cases_satisfying_alternate_precedence))
        print("Cases violating AlternatePrecedence (B occurs without proper A):", len(cases_with_second_event.difference(cases_satisfying_alternate_precedence)))
        cases_violating_alternate_precedence = cases_with_second_event.difference(cases_satisfying_alternate_precedence)

        #Add outcome column with Yes for cases that satisfy AlternatePrecedence and No for cases that violate it
        df["Outcome"] = df["case:concept:name"].apply(lambda x: "Satisfied" if x in cases_satisfying_alternate_precedence else ("Violated" if x in cases_violating_alternate_precedence else "Unknown"))
        # Get cases that have the second event
        cases_with_second_event = set(df[df["concept:name"].isin(second_events)]["case:concept:name"].unique())
        # Get the first row where the second event occurs for each case
        df = (
        df[df["case:concept:name"].isin(cases_with_second_event) & df["concept:name"].isin(second_events)]
            .groupby("case:concept:name", as_index=False)   
            .head(1)    
        )
        print("Dataframe shape:", df.shape)

        # Drop some columns that are not relevant for the analysis
        df = df.drop(columns=['concept:name', 'time:timestamp', 'case:concept:name','case:Item Category','User','case:GR-Based Inv. Verif.','case:Goods Receipt','case:Document Type','case:Name','case:Vendor','case:Source','org:resource','case:Purch. Doc. Category name','case:Purchasing Document','case:Company'])

        
        # Now we have a DataFrame with only the Outcome column and the case identifier. We can proceed to calculate the Shapley values based on this DataFrame.
        shap_values = conditional_shapley_with_binning(df)

        print("\nConditional Shapley values are as follows:")
        for f,v in shap_values.items():
            print(f"{f}: {round(v,4)}")

        # Prepare the data for decision tree building
        # Balance the dataset by taking all "Violated" cases and an equal number of "Satisfied" cases
        print("\nNow we are building the decision tree")
        print("Dataframe shape before balancing:", df.shape)
        print("Number of cases with Outcome = Violated:", (df["Outcome"] == "Violated").sum())
        print("Number of cases with Outcome = Satisfied:", (df["Outcome"] == "Satisfied").sum())

        # Check if there are enough cases to build a decision tree
        if (df["Outcome"] == "Violated").sum() == 0 or (df["Outcome"] == "Satisfied").sum() == 0:
            print("Not enough cases to build a decision tree. Skipping decision tree building.")
            return
      

        # Balance the dataset by taking all "Violated" cases and an equal number of "Satisfied" cases
        df_violated = df[df["Outcome"] == "Violated"]
        # If there are more "Satisfied" cases than "Violated" cases, sample an equal number of "Satisfied" cases
        if len(df[df["Outcome"] == "Satisfied"]) > len(df_violated):
            while True:
                print("\nHow would you like to sample the 'Satisfied' cases to balance the dataset?")
                print("1. Random sample of 'Satisfied' cases equal to the number of 'Violated' cases:")
                print("2. Sample from beginning of 'Satisfied' cases equal to the number of 'Violated' cases:")
                print("3. Sample from end of 'Satisfied' cases equal to the number of 'Violated' cases:")
                selected_sample_type=input(f"Enter your choice (1-3): ")
                if selected_sample_type in ["1", "2", "3"]:    
                    if selected_sample_type == "1":
                        df_satisfied = df[df["Outcome"] == "Satisfied"].sample(n=len(df_violated), random_state=42)
                    elif selected_sample_type == "2":
                        df_satisfied = df[df["Outcome"] == "Satisfied"].head(len(df_violated))
                    elif selected_sample_type == "3":
                        df_satisfied = df[df["Outcome"] == "Satisfied"].tail(len(df_violated))
                else:
                    print("Invalid choice. Please try again.")
                    # Continue the loop to ask for input again for invalid choice
                    continue
                # Break the loop if a valid choice was made
                break
        else:
            df_satisfied = df[df["Outcome"] == "Satisfied"]
        prepared_df = pd.concat([df_violated, df_satisfied])
        print("Dataframe shape after balancing:", prepared_df.shape)

        build_decison_trees_function(constraint['id'],prepared_df)

    else:
        print(f"Activity: {constraint['activity']}")
        # In case of Unary constraints, we can have different templates like AtMostOne, End, etc. We will handle them accordingly.
        if constraint.get("template") == "AtMostOne":
            print(f"Calculating Shapley value for {constraint['template']} constraint...")
            first_events = constraint['activity']
            # Create a copy of the event logs DataFrame to avoid modifying the original
            df=event_logs.copy()
            grouped = df.groupby("case:concept:name")["concept:name"].agg(list)
            # AtMostOne
            cases_satisfying_at_most_one = grouped[grouped.apply(
                lambda x: has_at_most_one(x, A_list=first_events)
            )].index

            # Count breakdown
            print("Total cases:", len(grouped))
            print("Cases satisfying AtMostOne:", len(cases_satisfying_at_most_one))
            print("Cases violating AtMostOne:", len(grouped) - len(cases_satisfying_at_most_one))
            cases_violating_at_most_one = grouped[~grouped.apply(
                lambda x: has_at_most_one(x, A_list=first_events)
            )].index

            # Add outcome column with Satisfied for cases that satisfy AtMostOne and Violated for cases that violate it
            df["Outcome"] = df["case:concept:name"].apply(lambda x: "Satisfied" if x in cases_satisfying_at_most_one else ("Violated" if x in cases_violating_at_most_one else "Unknown"))
            # Get first row for each case where the event in first_events occurs
            cases_with_first_event = set(df[df["concept:name"].isin(first_events)]["case:concept:name"].unique())
            df = (
                df[df["case:concept:name"].isin(cases_with_first_event) & df["concept:name"].isin(first_events)]
                .groupby("case:concept:name", as_index=False)
                .head(1)
            )
            print("Dataframe shape:", df.shape)
            # Drop some columns that are not relevant for the analysis
            df = df.drop(columns=['concept:name', 'time:timestamp', 'case:concept:name','case:Item Category','User','case:GR-Based Inv. Verif.','case:Goods Receipt','case:Document Type','case:Name','case:Vendor','case:Source','org:resource','case:Purch. Doc. Category name','case:Purchasing Document','case:Company'])
            # Display the first few rows of the DataFrame to verify the structure before calculating Shapley values
        
            # Now we have a DataFrame with only the Outcome column and the case identifier. We can proceed to calculate the Shapley values based on this DataFrame.
            shap_values = conditional_shapley_with_binning(df)

            print("\nConditional Shapley values are as follows:")
            for f,v in shap_values.items():
                print(f"{f}: {round(v,4)}")

            # Prepare the data for decision tree building
            # Balance the dataset by taking all "Violated" cases and an equal number of "Satisfied" cases
            print("\nNow we are building the decision tree")
            print("Dataframe shape before balancing:", df.shape)
            print("Number of cases with Outcome = Violated:", (df["Outcome"] == "Violated").sum())
            print("Number of cases with Outcome = Satisfied:", (df["Outcome"] == "Satisfied").sum())

            # Check if there are enough cases to build a decision tree
            if (df["Outcome"] == "Violated").sum() == 0 or (df["Outcome"] == "Satisfied").sum() == 0:
                print("Not enough cases to build a decision tree. Skipping decision tree building.")
                return
        
            # Balance the dataset by taking all "Violated" cases and an equal number of "Satisfied" cases
            df_violated = df[df["Outcome"] == "Violated"]
            # If there are more "Satisfied" cases than "Violated" cases, sample an equal number of "Satisfied" cases
            if len(df[df["Outcome"] == "Satisfied"]) > len(df_violated):
                while True:
                    print("\nHow would you like to sample the 'Satisfied' cases to balance the dataset?")
                    print("1. Random sample of 'Satisfied' cases equal to the number of 'Violated' cases:")
                    print("2. Sample from beginning of 'Satisfied' cases equal to the number of 'Violated' cases:")
                    print("3. Sample from end of 'Satisfied' cases equal to the number of 'Violated' cases:")
                    selected_sample_type=input(f"Enter your choice (1-3): ")
                    if selected_sample_type in ["1", "2", "3"]:    
                        if selected_sample_type == "1":
                            df_satisfied = df[df["Outcome"] == "Satisfied"].sample(n=len(df_violated), random_state=42)
                        elif selected_sample_type == "2":
                            df_satisfied = df[df["Outcome"] == "Satisfied"].head(len(df_violated))
                        elif selected_sample_type == "3":
                            df_satisfied = df[df["Outcome"] == "Satisfied"].tail(len(df_violated))
                    else:
                        print("Invalid choice. Please try again.")
                        # Continue the loop to ask for input again for invalid choice
                        continue
                    # Break the loop if a valid choice was made
                    break
            else:
                df_satisfied = df[df["Outcome"] == "Satisfied"]
            prepared_df = pd.concat([df_violated, df_satisfied])
            print("Dataframe shape after balancing:", prepared_df.shape)

            build_decison_trees_function(constraint['id'],prepared_df)    
        
        else:
            print(f"Calculating Shapley value for {constraint['template']} constraint...")
            first_events = constraint['activity']
            # Create a copy of the event logs DataFrame to avoid modifying the original
            df=event_logs.copy()
            grouped = df.groupby("case:concept:name")["concept:name"].agg(list)
            # End
            cases_satisfying_end = grouped[grouped.apply(
                lambda x: has_end(x, A_list=first_events)
            )].index

            # Count breakdown
            print("Total cases:", len(grouped))
            print("Cases satisfying End:", len(cases_satisfying_end))
            print("Cases violating End:", len(grouped) - len(cases_satisfying_end))
            cases_violating_end = grouped[~grouped.apply(
                lambda x: has_end(x, A_list=first_events)
            )].index
            
            # Add outcome column with Satisfied for cases that satisfy End constraint and Violated for cases that violate it
            df["Outcome"] = df["case:concept:name"].apply(lambda x: "Satisfied" if x in cases_satisfying_end else ("Violated" if x in cases_violating_end else "Unknown"))
            df = df.drop_duplicates(subset=['case:concept:name'])
             
            print("Dataframe shape:", df.shape)

            # Drop some columns that are not relevant for the analysis
            df = df.drop(columns=['concept:name', 'time:timestamp', 'case:concept:name','case:Item Category','User','case:GR-Based Inv. Verif.','case:Goods Receipt','case:Document Type','case:Name','case:Vendor','case:Source','org:resource','case:Purch. Doc. Category name','case:Purchasing Document','case:Company'])
            # Unique cases with case:concept:name           
           

            # Now we have a DataFrame with only the Outcome column and the case identifier. We can proceed to calculate the Shapley values based on this DataFrame.
            shap_values = conditional_shapley_with_binning(df)

            print("\nConditional Shapley values are as follows:")
            for f,v in shap_values.items():
                print(f"{f}: {round(v,4)}")

            # Prepare the data for decision tree building
            # Balance the dataset by taking all "Violated" cases and an equal number of "Satisfied" cases
            print("\nNow we are building the decision tree")
            print("Dataframe shape before balancing:", df.shape)
            print("Number of cases with Outcome = Violated:", (df["Outcome"] == "Violated").sum())
            print("Number of cases with Outcome = Satisfied:", (df["Outcome"] == "Satisfied").sum())

            # Check if there are enough cases to build a decision tree
            if (df["Outcome"] == "Violated").sum() == 0 or (df["Outcome"] == "Satisfied").sum() == 0:
                print("Not enough cases to build a decision tree. Skipping decision tree building.")
                return
            
            # Balance the dataset by taking all "Violated" cases and an equal number of "Satisfied" cases
            df_violated = df[df["Outcome"] == "Violated"]
            # If there are more "Satisfied" cases than "Violated" cases, sample an equal number of "Satisfied" cases
            if len(df[df["Outcome"] == "Satisfied"]) > len(df_violated):
                while True:
                    print("\nHow would you like to sample the 'Satisfied' cases to balance the dataset?")
                    print("1. Random sample of 'Satisfied' cases equal to the number of 'Violated' cases:")
                    print("2. Sample from beginning of 'Satisfied' cases equal to the number of 'Violated' cases:")
                    print("3. Sample from end of 'Satisfied' cases equal to the number of 'Violated' cases:")
                    selected_sample_type=input(f"Enter your choice (1-3): ")
                    if selected_sample_type in ["1", "2", "3"]:    
                        if selected_sample_type == "1":
                            df_satisfied = df[df["Outcome"] == "Satisfied"].sample(n=len(df_violated), random_state=42)
                        elif selected_sample_type == "2":
                            df_satisfied = df[df["Outcome"] == "Satisfied"].head(len(df_violated))
                        elif selected_sample_type == "3":
                            df_satisfied = df[df["Outcome"] == "Satisfied"].tail(len(df_violated))
                    else:
                        print("Invalid choice. Please try again.")
                        # Continue the loop to ask for input again for invalid choice
                        continue
                    # Break the loop if a valid choice was made
                    break
            else:
                df_satisfied = df[df["Outcome"] == "Satisfied"]
            prepared_df = pd.concat([df_violated, df_satisfied])
            print("Dataframe shape after balancing:", prepared_df.shape)

            build_decison_trees_function(constraint['id'],prepared_df)
