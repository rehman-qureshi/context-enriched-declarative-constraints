import csv
from datetime import datetime
import pandas as pd
from find_shapley_values import conditional_shapley_with_binning
from build_decision_trees import build_decision_trees_function
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
def check_conformance_function(event_logs, constraints,selected_option):
    
    # selected_option is either '1' for a particular constraint analysis or '2' for integrated analysis of all constraints
    if selected_option == '1':
            # Assuming constraints is a single constraint dictionary when selected_option is '1'
            constraint = constraints
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
                print("\nConditional Shapley values with respect to the violated case are as follows:")
                # Global baseline for the violated case is the probability of violation without any features, which is the Outcome column being "Violated" divided by the total number of activated cases (i.e., cases that have the second event). This is calculated as follows:
                global_baseline = len(cases_violating_alternate_precedence)/len(cases_with_second_event)
                print(f"Global baseline for the violated case: {round(global_baseline,4)}")
                for f,v in shap_values.items():
                    contribution = (v/global_baseline)*100 if global_baseline != 0 else float('inf')  # Avoid division by zero
                    print(f"{f}: {round(v,4)}, contribution: {round(contribution,2)}%")

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

                # Initialize the activation_conditions variable to store the result of decision tree building
                activation_conditions = None
                activation_conditions = build_decision_trees_function(constraint['id'],prepared_df)
                if activation_conditions:
                    print(f"\nActivation conditions ({len(activation_conditions)}):")
                    # Add a counter to enumerate the activation conditions for better readability
                    for idx, condition in enumerate(activation_conditions, start=1):
                        print(f"{idx}. {condition}")
                else:
                    activation_conditions = [f"No satisfied leaf found for constraint {constraint_id}"]

            # In case of Unary constraints, we can have different templates like AtMostOne, End, etc. We will handle them accordingly.
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

                    print("\nConditional Shapley values with respect to the violated case are as follows:")
                    # Global baseline for the violated case is the probability of violation without any features, which is the number of violated cases divided by the total number of activated cases (i.e., len(grouped)). This is calculated as follows:
                    global_baseline = len(cases_violating_at_most_one)/len(grouped)
                    print(f"Global baseline for the violated case: {round(global_baseline,4)}")
                    for f,v in shap_values.items():
                        contribution = (v/global_baseline)*100 if global_baseline != 0 else float('inf')  # Avoid division by zero
                        print(f"{f}: {round(v,4)}, contribution: {round(contribution,2)}%")

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

                    # Initialize the activation_conditions variable to store the result of decision tree building
                    activation_conditions = None
                    activation_conditions = build_decision_trees_function(constraint['id'],prepared_df)
                    if activation_conditions:
                        print(f"\nActivation conditions ({len(activation_conditions)}):")
                        # Add a counter to enumerate the activation conditions for better readability
                        for idx, condition in enumerate(activation_conditions, start=1):
                            print(f"{idx}. {condition}")
                    else:
                        activation_conditions = [f"No satisfied leaf found for constraint {constraint_id}"]
                
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
                

                    # Now we have a DataFrame with only the Outcome column and the case identifier. We can proceed to calculate the Shapley values based on this DataFrame.
                    shap_values = conditional_shapley_with_binning(df)

                    print("\nConditional Shapley values with respect to the violated case are as follows:")

                    # Global baseline for the violated case is the probability of violation without any features, which is the number of violated cases divided by the total number of activated cases (i.e., len(grouped)). This is calculated as follows:
                    global_baseline = len(cases_violating_end)/len(grouped)
                    print(f"Global baseline for the violated case: {round(global_baseline,4)}")
                    for f,v in shap_values.items():
                        contribution = (v/global_baseline)*100 if global_baseline != 0 else float('inf')  # Avoid division by zero
                        print(f"{f}: {round(v,4)}, contribution: {round(contribution,2)}%")

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
                    # Initialize the activation_condition variable to store the result of decision tree building
                    activation_conditions = None
                    activation_conditions = build_decision_trees_function(constraint['id'],prepared_df)
                    if activation_conditions:
                        print(f"\nActivation conditions ({len(activation_conditions)}):")
                        # Add a counter to enumerate the activation conditions for better readability
                        for idx, condition in enumerate(activation_conditions, start=1):
                            print(f"{idx}. {condition}")
                    else:
                        activation_conditions = [f"No satisfied leaf found for constraint {constraint_id}"]
    
    elif selected_option == '2':
        print("\nIntegrated analysis experience on the most promising constraints is started.")
        # Here we can add code to perform integrated analysis based on the most promising constraints 
        conformance_results = {}  
        print("If conformance rate is less than 0.98, we will proceed to calculate Shapley values and build decision trees for that constraint, otherwise we will skip it.")
        for constraint in constraints:
            print(f"\nStarting analysis for constraint: {constraint['id']} - {constraint['template']}")
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
                # Compute the conformance rate for the constraint
                conformance_rate=len(cases_satisfying_alternate_precedence)/len(cases_with_second_event) if len(cases_with_second_event) > 0 else 0
                print(f"Conformance rate for constraint {constraint['id']} - {constraint['template']}: {round(conformance_rate,4)}")
                # If conformance rate is less than 0.98, we will proceed to calculate Shapley values and build decision trees for this constraint, otherwise we will skip it.
                if conformance_rate < 0.98:
                    # Create a new entry in the conformance_results dictionary for this constraint if it doesn't exist
                    conformance_results.setdefault(constraint["id"], {})
                    # Save the conformance rate for this constraint in the conformance_results dictionary
                    conformance_results[constraint['id']]["conformance_rate"] = conformance_rate
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
                    print("\nConditional Shapley values with respect to the violated case are as follows:")
                    # Global baseline for the violated case is the probability of violation without any features, which is the Outcome column being "Violated" divided by the total number of activated cases (i.e., cases that have the second event). This is calculated as follows:
                    global_baseline = len(cases_violating_alternate_precedence)/len(cases_with_second_event) if len(cases_with_second_event) > 0 else 0
                    print(f"Global baseline for the violated case: {round(global_baseline,4)}")
                    contribution_dict = {}
                    for f,v in shap_values.items():
                        contribution = (v/global_baseline)*100 if global_baseline != 0 else float('inf')  # Avoid division by zero
                        print(f"{f}: {round(v,4)}, contribution: {round(contribution,2)}%")
                        # Store the contribution in the dictionary for later use
                        contribution_dict[f] = round(contribution,2)

                    # Store the Shapley values and contributions in the conformance_results dictionary for this constraint
                    conformance_results[constraint['id']]["shapley_contributions"]=contribution_dict
                    # Reset the contribution_dict for the next constraint
                    contribution_dict = {}
                    # Prepare the data for decision tree building
                    # Balance the dataset by taking all "Violated" cases and an equal number of "Satisfied" cases
                    print("\nNow we are building the decision tree")
                    print("Dataframe shape before balancing:", df.shape)
                    print("Number of cases with Outcome = Violated:", (df["Outcome"] == "Violated").sum())
                    print("Number of cases with Outcome = Satisfied:", (df["Outcome"] == "Satisfied").sum())

                    # Check if there are enough cases to build a decision tree
                    if (df["Outcome"] == "Violated").sum() == 0 or (df["Outcome"] == "Satisfied").sum() == 0:
                        print("Not enough cases to build a decision tree. Skipping decision tree building.")
                        continue  # Skip to the next constraint in the loop

                    # Balance the dataset by taking all "Violated" cases and an equal number of "Satisfied" cases
                    df_violated = df[df["Outcome"] == "Violated"]
                    # If there are more "Satisfied" cases than "Violated" cases, sample an equal number of "Satisfied" cases
                    if len(df[df["Outcome"] == "Satisfied"]) > len(df_violated):
                        # Random sample is selected for integrated analysis, so we will sample an equal number of 'Satisfied' cases to match the number of 'Violated' cases.
                        print("Random sample is selected for integrated analysis, so we will sample an equal number of 'Satisfied' cases to match the number of 'Violated' cases.")
                        df_satisfied = df[df["Outcome"] == "Satisfied"].sample(n=len(df_violated), random_state=42)
                    else:
                        df_satisfied = df[df["Outcome"] == "Satisfied"]
                    prepared_df = pd.concat([df_violated, df_satisfied])
                    print("Dataframe shape after balancing:", prepared_df.shape)

                    # Build decision trees for this constraint and store the activation condition in the conformance_results dictionary
                    conformance_results[constraint['id']]["activation_conditions"] = build_decision_trees_function(constraint['id'],prepared_df)
                
                else:
                    print(f"Conformance rate for constraint {constraint['id']} - {constraint['template']}: {round(conformance_rate,4)}")
                    print(f"Skipping Shapley value calculation and decision tree building for constraint {constraint['id']} - {constraint['template']} due to conformance rate >= 0.98.")
           
            # In case of Unary constraints, we can have different templates like AtMostOne, End, etc. We will handle them accordingly.
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
                    
                    # Compute the conformance rate for the constraint
                    conformance_rate=len(cases_satisfying_at_most_one)/len(grouped) if len(grouped) > 0 else 0
                    print(f"Conformance rate for constraint {constraint['id']} - {constraint['template']}: {round(conformance_rate,4)}")
                    # If conformance rate is less than 0.98, we will proceed to calculate
                    if conformance_rate < 0.98:
                        # Create a new entry in the conformance_results dictionary for this constraint if it doesn't exist 
                        conformance_results.setdefault(constraint["id"], {})
                        # Save the conformance rate for this constraint in the conformance_results dictionary
                        conformance_results[constraint['id']]["conformance_rate"] = conformance_rate
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

                        print("\nConditional Shapley values with respect to the violated case are as follows:")
                        # Global baseline for the violated case is the probability of violation without any features, which is the number of violated cases divided by the total number of activated cases (i.e., len(grouped)). This is calculated as follows:
                        global_baseline = len(cases_violating_at_most_one)/len(grouped) if len(grouped) > 0 else 0
                        print(f"Global baseline for the violated case: {round(global_baseline,4)}")
                        contribution_dict = {}
                        for f,v in shap_values.items():
                            contribution = (v/global_baseline)*100 if global_baseline != 0 else float('inf')  # Avoid division by zero
                            print(f"{f}: {round(v,4)}, contribution: {round(contribution,2)}%")
                            # Store the contribution in the dictionary for later use
                            contribution_dict[f] = contribution
                            
                        # Store the Shapley values and contributions in the conformance_results dictionary for this constraint
                        conformance_results[constraint['id']]["shapley_contributions"]=contribution_dict
                        # Reset the contribution_dict for the next constraint
                        contribution_dict = {}
                        # Prepare the data for decision tree building
                        # Balance the dataset by taking all "Violated" cases and an equal number of "Satisfied" cases
                        print("\nNow we are building the decision tree")
                        print("Dataframe shape before balancing:", df.shape)
                        print("Number of cases with Outcome = Violated:", (df["Outcome"] == "Violated").sum())
                        print("Number of cases with Outcome = Satisfied:", (df["Outcome"] == "Satisfied").sum())

                        # Check if there are enough cases to build a decision tree
                        if (df["Outcome"] == "Violated").sum() == 0 or (df["Outcome"] == "Satisfied").sum() == 0:
                            print("Not enough cases to build a decision tree. Skipping decision tree building.")
                            continue  # Skip to the next constraint if there are not enough cases
                    
                        # Balance the dataset by taking all "Violated" cases and an equal number of "Satisfied" cases
                        df_violated = df[df["Outcome"] == "Violated"]
                        # If there are more "Satisfied" cases than "Violated" cases, sample an equal number of "Satisfied" cases
                        if len(df[df["Outcome"] == "Satisfied"]) > len(df_violated):
                            # Random sample is selected for integrated analysis, so we will sample an equal number of "Satisfied" cases to match the number of "Violated" cases
                            print("Random sample is selected for integrated analysis, so we will sample an equal number of 'Satisfied' cases to match the number of 'Violated' cases.")
                            df_satisfied = df[df["Outcome"] == "Satisfied"].sample(n=len(df_violated), random_state=42)                
                        else:
                            df_satisfied = df[df["Outcome"] == "Satisfied"]
                        prepared_df = pd.concat([df_violated, df_satisfied])
                        print("Dataframe shape after balancing:", prepared_df.shape)

                        # Build decision trees for this constraint and store the activation condition in the conformance_results dictionary
                        conformance_results[constraint['id']]["activation_conditions"] = build_decision_trees_function(constraint['id'],prepared_df) 
                
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
                    
                    # Compute the conformance rate for the constraint
                    conformance_rate=len(cases_satisfying_end)/len(grouped) if len(grouped) > 0 else 0
                    print(f"Conformance rate for constraint {constraint['id']} - {constraint['template']}: {round(conformance_rate,4)}")
                    # If conformance rate is less than 0.98, we will proceed to calculate
                    if conformance_rate < 0.98:
                        # Create a new dictionary for this constraint if it doesn't exist yet
                        conformance_results.setdefault(constraint["id"], {})
                        # Save the conformance rate for this constraint in the conformance_results dictionary
                        conformance_results[constraint['id']]['conformance_rate'] = conformance_rate
                        # Add outcome column with Satisfied for cases that satisfy End constraint and Violated for cases that violate it
                        df["Outcome"] = df["case:concept:name"].apply(lambda x: "Satisfied" if x in cases_satisfying_end else ("Violated" if x in cases_violating_end else "Unknown"))
                        df = df.drop_duplicates(subset=['case:concept:name'])
                        
                        print("Dataframe shape:", df.shape)

                        # Drop some columns that are not relevant for the analysis
                        df = df.drop(columns=['concept:name', 'time:timestamp', 'case:concept:name','case:Item Category','User','case:GR-Based Inv. Verif.','case:Goods Receipt','case:Document Type','case:Name','case:Vendor','case:Source','org:resource','case:Purch. Doc. Category name','case:Purchasing Document','case:Company'])
                        # Unique cases with case:concept:name           
                    
                        # Now we have a DataFrame with only the Outcome column and the case identifier. We can proceed to calculate the Shapley values based on this DataFrame.
                        shap_values = conditional_shapley_with_binning(df)

                        print("\nConditional Shapley values with respect to the violated case are as follows:")

                        # Global baseline for the violated case is the probability of violation without any features, which is the number of violated cases divided by the total number of activated cases (i.e., len(grouped)). This is calculated as follows:
                        global_baseline = len(cases_violating_end)/len(grouped)
                        print(f"Global baseline for the violated case: {round(global_baseline,4)}")
                        contribution_dict = {}
                        for f,v in shap_values.items():
                            contribution = (v/global_baseline)*100 if global_baseline != 0 else float('inf')  # Avoid division by zero
                            print(f"{f}: {round(v,4)}, contribution: {round(contribution,2)}%")
                            # Store the contribution in the dictionary for later use
                            contribution_dict[f] = contribution

                        # Store the Shapley values and contributions in the conformance_results dictionary for this constraint
                        # Create a new dictionary for this constraint if it doesn't exist yet
                        conformance_results.setdefault(constraint["id"], {})
                        conformance_results[constraint['id']]["shapley_contributions"]=contribution_dict
                        # Reset the contribution_dict for the next constraint
                        contribution_dict = {}
                        # Prepare the data for decision tree building
                        # Balance the dataset by taking all "Violated" cases and an equal number of "Satisfied" cases
                        print("\nNow we are building the decision tree")
                        print("Dataframe shape before balancing:", df.shape)
                        print("Number of cases with Outcome = Violated:", (df["Outcome"] == "Violated").sum())
                        print("Number of cases with Outcome = Satisfied:", (df["Outcome"] == "Satisfied").sum())

                        # Check if there are enough cases to build a decision tree
                        if (df["Outcome"] == "Violated").sum() == 0 or (df["Outcome"] == "Satisfied").sum() == 0:
                            print("Not enough cases to build a decision tree. Skipping decision tree building.")
                            continue # Skip to the next constraint if there are not enough cases
                        
                        # Balance the dataset by taking all "Violated" cases and an equal number of "Satisfied" cases
                        df_violated = df[df["Outcome"] == "Violated"]
                        # If there are more "Satisfied" cases than "Violated" cases, sample an equal number of "Satisfied" cases
                        if len(df[df["Outcome"] == "Satisfied"]) > len(df_violated):
                            print("Random sample is selected for integrated analysis, so we will sample an equal number of 'Satisfied' cases to match the number of 'Violated' cases.")
                            df_satisfied = df[df["Outcome"] == "Satisfied"].sample(n=len(df_violated), random_state=42)
                        else:
                            df_satisfied = df[df["Outcome"] == "Satisfied"]
                        prepared_df = pd.concat([df_violated, df_satisfied])
                        print("Dataframe shape after balancing:", prepared_df.shape)
                        # Build decision trees for this constraint and store the activation condition in the conformance_results dictionary
                        conformance_results[constraint['id']]["activation_conditions"] = build_decision_trees_function(constraint['id'],prepared_df)

        print("\nIntegrated analysis experience of most promising constraints is completed.")
        print("We compute the Shapley values and build decision trees for constraints with conformance rate less than 0.98, and skip those with conformance rate >= 0.98.")
        # Rank the constraints based from the lowest to the highest conformance rate
        ranked_constraints = sorted(conformance_results.items(), key=lambda x: x[1]['conformance_rate'])
        print("Constraints ranked from lowest to highest conformance rate:")
        for constraint_id, conformance_rate in ranked_constraints:
            # We consider a conformance rate of less than 0.98 as a threshold for further analysis, so we will only print those constraints that have a conformance rate below this threshold.
            if conformance_rate['conformance_rate'] < 0.98:
                # find constraint from constraints list using constraint_id
                constraint = next((c for c in constraints if c['id'] == constraint_id), None)
                if constraint:
                    if constraint["type"] == "Binary":
                        #print(f"{constraint_id}: {constraint['template']} (Source: {constraint['source']}, Target: {constraint['target']}): {round(conformance_rate,4)}")
                        # \033[1m {variable} \003[0m is used to make the text bold in the terminal output
                         print(f"{constraint_id}. {constraint['template']}({{{', '.join(constraint['source'])}}}, {{{', '.join(constraint['target'])}}}): \033[1m{round(conformance_rate['conformance_rate'],4)}\033[0m")
                         # Display the activation conditions for this constraint
                         activation_conditions = conformance_rate.get('activation_conditions', [])
                         if activation_conditions:
                            print(f"\nActivation conditions for constraint {constraint_id}:")
                            for idx, condition in enumerate(activation_conditions, start=1):
                                 print(f"{idx}. {condition}")
                         else:
                             print(f"No activation conditions found for constraint {constraint_id}.")
                         # Display the Shapley contributions for this constraint
                         print(f"\nShapley contributions towards violated cases for constraint {constraint_id}:")
                         for f, contribution in conformance_rate['shapley_contributions'].items():
                             print(f"  {f}: {round(contribution,2)}%")                                                
                    else:
                        # For Unary constraints, we can have different templates like AtMostOne, End, etc. We will handle them accordingly.
                        #print(f"{constraint_id}: {constraint['template']} (Activity: {constraint['activity']}): {round(conformance_rate,4)}")
                        print(f"{constraint['id']}. {constraint['template']}({{{', '.join(constraint['activity'])}}}): \033[1m{round(conformance_rate['conformance_rate'],4)}\033[0m")
                        # Display the activation conditions for this constraint
                        activation_conditions = conformance_rate.get('activation_conditions', [])
                        if activation_conditions:
                            print(f"\nActivation conditions for constraint {constraint_id}:")
                            for idx, condition in enumerate(activation_conditions, start=1):
                                print(f"{idx}. {condition}")
                        else:
                            print(f"No activation conditions found for constraint {constraint_id}.")
                        # Display the Shapley contributions for this constraint
                        print(f"\nShapley contributions towards violated cases for constraint {constraint_id}:")
                        for f, contribution in conformance_rate['shapley_contributions'].items():
                            print(f"  {f}: {round(contribution,2)}%")
            # Also store integrated experience as csv file for further analysis
            filename = f"integrated_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Constraint ID', 'Constraint', 'Conformance Rate', 'Activation Conditions', 'Shapley Contributions Toward Violated Cases'])
                for constraint_id, conformance_rate in ranked_constraints:
                    # We consider a conformance rate of less than 0.98 as a threshold for further analysis, so we will only write those constraints that have a conformance rate below this threshold.
                    if conformance_rate['conformance_rate'] < 0.98:
                        # find constraint from constraints list using constraint_id
                        constraint = next((c for c in constraints if c['id'] == constraint_id), None)
                        if constraint:
                            activation_conditions = conformance_rate.get('activation_conditions', [])
                            shapley_contributions = conformance_rate.get('shapley_contributions', {})
                            constraint_format=""
                            # Display each constraint in a desired format for csv report
                            if constraint.get("type")=="Unary":
                                constraint_format=f"{constraint['template']}({{{', '.join(constraint['activity'])}}}) "
                            else:
                                constraint_format=f"{constraint['template']}({{{', '.join(constraint['source'])}}}, {{{', '.join(constraint['target'])}}}) "
                            writer.writerow([
                                constraint_id,
                                constraint_format,
                                round(conformance_rate['conformance_rate'],4),
                                "; ".join(activation_conditions),
                                "; ".join([f"{f}: {round(contribution,2)}%" for f, contribution in shapley_contributions.items()])
                            ])
                writer.writerow([])  # Add an empty row for better readability
                writer.writerow(['Note: Only constraints with conformance rate less than 0.98 are included in this report.'])
                print(f"Integrated analysis experience is exported as {filename}.")
    else:
        print("Wrong option selected.") # Not possible to reach here because of the input validation in driver.py

