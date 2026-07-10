
import sys
from load_data import load_data_function 
from check_conformance import check_conformance_function
import warnings

# Ignore warnings related to rustxes when reading XES files
warnings.filterwarnings(
    "ignore",
    message=r".*rustxes.*"
)
# Main function to load data and check conformance
if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python driver.py <events_logs_path> <declarative_constraints_path>")
        sys.exit(1)

    events_logs_path = sys.argv[1]
    declarative_constraints_path = sys.argv[2]


    event_logs, declarative_constraints = load_data_function(events_logs_path, declarative_constraints_path)


    # Display declarative constraints so that user can select which constraints to use for the analysis.
    if declarative_constraints:
        print("\nDeclarative Constraints:")
        for constraint in declarative_constraints:
            if constraint.get("type")=="Unary":
             print(f"{constraint['id']}. {constraint['template']}({{{', '.join(constraint['activity'])}}}) ")
            else:
             print(f"{constraint['id']}. {constraint['template']}({{{', '.join(constraint['source'])}}}, {{{', '.join(constraint['target'])}}}) ")
    
    while True:
        print("\n1. Select a particular constraint by its ID to for anaylsis.")
        print("2. Integrated analysis experience of most promising constraints.")
        print("3. Exit the program.")
        selected_option = input("\nEnter your choice (1, 2, or 3): ")
        if selected_option == '1':
            selected_constraint=input("\nEnter the ID of the constraint you want to use for analysis: ")
            #if selected_constraint.lower() == 'exit':
                #print("Exiting the program.")
                #break
            # Check if the selected constraint ID is valid
            if any(constraint['id'] == selected_constraint for constraint in declarative_constraints):
                print(f"You selected constraint ID: {selected_constraint}")
                # Here we can add code to perform analysis based on the selected constraint
                check_conformance_function(event_logs, next(constraint for constraint in declarative_constraints if constraint['id'] == selected_constraint),selected_option)
                #print(f"Shapley Value for Constraint {selected_constraint}: {shapley_value}")
            else:
                print("Invalid constraint ID. Please try again.")            

        elif selected_option == '2':
            # Here we can add code to perform integrated analysis based on the most promising constraints
            check_conformance_function(event_logs, declarative_constraints,selected_option)
        elif selected_option == '3':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")