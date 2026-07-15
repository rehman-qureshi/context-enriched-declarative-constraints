# Loading events logs from XES file and declarative constraints from JSON file

import json
import pm4py
import pandas as pd
from io import StringIO

#----------------------------------------
def load_events_logs(file_path):
    # Read XES file
    """LOG = pm4py.read_xes(file_path)
    # ALWAYS convert explicitly
    df = pm4py.convert_to_dataframe(LOG)
    # sort:
    # 1) by case identifier (ascending)
    # 2) by event timestamp (ascending within each case)
    df = df.sort_values(
        by=["case:concept:name", "time:timestamp"],
        ascending=[True, True]
    ).copy()

    # If df is empty, raise an error
    if df.empty:
        raise ValueError("The events logs dataframe is empty. Please check the input file.")
    
    # Convert the dataframe to CSV format in memory using StringIO
    buffer = StringIO()

    df.to_csv(buffer, index=False)
    buffer.seek(0)

    df = pd.read_csv(buffer)"""
    
    # To save processing time, we read the preprocessed CSV file instead of the XES file. The CSV file is assumed to be in the same directory as this script.
    df = pd.read_csv("bpic19_invoice_before_gr_all_cases.csv")

    return df
#-------------------------------------------
def load_declarative_constraints(file_path):

    # Read the declarative_constraints.json file and return the constraints as a list of strings
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data["constraints"]
#-------------------------------------------
def load_data_function(events_logs_path, declarative_constraints_path):
   
    event_logs = load_events_logs(events_logs_path)
    declarative_constraints = load_declarative_constraints(declarative_constraints_path)

    return event_logs, declarative_constraints
#--------------------------------------------