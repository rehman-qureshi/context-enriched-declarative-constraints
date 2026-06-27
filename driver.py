
import sys

from load_data import load_data_function 
from check_conformance import check_conformance_function

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python driver.py <events_logs_path> <declarative_constraints_path>")
        sys.exit(1)

    events_logs_path = sys.argv[1]
    declarative_constraints_path = sys.argv[2]

    # Example usage
    # events_logs_path = "BPIC19_3way_IbeforeGR_standardPO_complete.xes"
    # declarative_constraints_path = "declarative_constraints.json"

    event_logs, declarative_constraints = load_data_function(events_logs_path, declarative_constraints_path)

    #print("Event Logs:")
    #print(event_logs.head())

    #print("\nDeclarative Constraints:")
    #print(declarative_constraints)
    # Display declarative constraints so that user can select which constraints to use for the analysis.

    if declarative_constraints:
        print("\nDeclarative Constraints:")
        for constraint in declarative_constraints:
            if constraint.get("type")=="Unary":
             print(f"{constraint['id']}. {constraint['template']}({{{', '.join(constraint['activity'])}}}) ")
            else:
             print(f"{constraint['id']}. {constraint['template']}({{{', '.join(constraint['source'])}}}, {{{', '.join(constraint['target'])}}}) ")

    while True:
        selected_constraint=input("\nEnter the ID of the constraint you want to use for analysis (or type 'exit' to quit): ")
        if selected_constraint.lower() == 'exit':
            print("Exiting the program.")
            break
        # Check if the selected constraint ID is valid
        if any(constraint['id'] == selected_constraint for constraint in declarative_constraints):
            print(f"You selected constraint ID: {selected_constraint}")
            # Here you can add code to perform analysis based on the selected constraint
            check_conformance_function(event_logs, next(constraint for constraint in declarative_constraints if constraint['id'] == selected_constraint))
            #print(f"Shapley Value for Constraint {selected_constraint}: {shapley_value}")
        else:
            print("Invalid constraint ID. Please try again.")            

       

   