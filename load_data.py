# Loading events logs from XES file and declarative constraints from JSON file

import json
from math import log
import pm4py
import pandas as pd
from io import StringIO

def load_events_logs(file_path):
    """
    Load events logs from a specified file path.

    Args:
        file_path (str): The path to the events logs file.

    Returns:
        pandas.DataFrame: A DataFrame containing the event logs.
    """
    # read XES file
    LOG = pm4py.read_xes(file_path)
    # ALWAYS convert explicitly
    df = pm4py.convert_to_dataframe(LOG)
    # sort:
    # 1) by case identifier (ascending)
    # 2) by event timestamp (ascending within each case)
    df = df.sort_values(
        by=["case:concept:name", "time:timestamp"],
        ascending=[True, True]
    ).copy()

    # if df is empty, raise an error
    if df.empty:
        raise ValueError("The events logs dataframe is empty. Please check the input file.")
    
    # convert the dataframe to CSV format in memory using StringIO
    buffer = StringIO()

    df.to_csv(buffer, index=False)
    buffer.seek(0)

    df = pd.read_csv(buffer)
    
    #df.to_csv("bpic19_invoice_before_gr_all_cases.csv", index=False)

    # To save processing time, we read the preprocessed CSV file instead of the XES file. The CSV file is assumed to be in the same directory as this script.
    #df = pd.read_csv("bpic19_invoice_before_gr_all_cases.csv")

    # export everything to CSV
    #df.to_csv("bpic19_invoice_before_gr_all_cases.csv", index=False)

    #print("Exported all cases to bpic19_invoice_before_gr_all_cases.csv")

    return df

def load_declarative_constraints(file_path):
    """
    Load declarative constraints from a specified file path.

    Args:
        file_path (str): The path to the declarative constraints file 

    Returns:
        list: A list of declarative constraints.
    """
    # read the declarative_constraints.json file and return the constraints as a list of strings

    with open(file_path, 'r') as file:
        data = json.load(file)
    return data["constraints"]

def load_data_function(events_logs_path, declarative_constraints_path):
    """
    Load events logs and declarative constraints from specified file paths.

    Args:
        events_logs_path (str): The path to the events logs file.
        declarative_constraints_path (str): The path to the declarative constraints file.

    Returns:
        tuple: A tuple containing the DataFrame of event logs and the list of declarative constraints.
    """
    event_logs = load_events_logs(events_logs_path)
    declarative_constraints = load_declarative_constraints(declarative_constraints_path)

    return event_logs, declarative_constraints