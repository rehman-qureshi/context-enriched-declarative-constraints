# Context-Enriched Declarative Constraints

This repository implements a Python workflow for analyzing declarative process constraints on BPIC19 purchase-to-pay event data. It supports interactive constraint selection, conformance evaluation, feature attribution with conditional Shapley values, and decision-tree-based explanations.

## What the project does

- Loads event-log data and declarative constraints.
- Evaluates unary and binary constraints such as AtMostOne, End, and AlternatePrecedence.
- Computes context-based feature importance using conditional Shapley values.
- Builds decision trees to explain whether cases satisfy or violate a chosen constraint.
- Exports the resulting decision-tree visualizations as PDF file.

## Repository structure

- [driver.py](driver.py) – Entry point. Loads the data, lists the available constraints, and asks the user to choose one.
- [load_data.py](load_data.py) – Loads event logs and declarative constraints from the repository inputs.
- [check_conformance.py](check_conformance.py) – Evaluates the selected constraint for each case and prepares the analysis dataset.
- [find_shapley_values.py](find_shapley_values.py) – Computes conditional Shapley values for the selected features.
- [build_decision_trees.py](build_decision_trees.py) – Builds a decision tree and exports a visualization.
- [declarative_constraints.json](declarative_constraints.json) – Example declarative constraints used by the workflow.
- [sabbrev_to_full.json](sabbrev_to_full.json) – Mapping file used for activity label normalization.
- [BPIC19_3way_IbeforeGR_standardPO_complete.xes](BPIC19_3way_IbeforeGR_standardPO_complete.xes) – Example event log used in the workflow.
- [requirements.txt](requirements.txt) – Python dependencies for the project.

### Prerequisites

- Python 3.x
- pandas
- pm4py
- sys
- Required [BPIC19_3way_IbeforeGR_standardPO_complete.xes](https://zenodo.org/records/17295283) file in the current folder. It contains event logs for the conformance checking and can be downloaded from [Zenado](https://zenodo.org/records/17295283).
- Requires a declarative constraints.json file.

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

The project depends on common Python packages such as pandas, pm4py, matplotlib, and graphviz.

## How to run

Run the main script as follows:

```bash
python driver.py <events_logs_path> <declarative_constraints_path>
```

Example:

```bash
python driver.py BPIC19_3way_IbeforeGR_standardPO_complete.xes declarative_constraints.json
```

After startup, the program prints the available constraints and prompts you to enter the ID of the constraint you want to analyze.

## Notes

- The workflow is interactive and prompts for constraint selection after loading the data.
- The analysis currently focuses on explaining constraint violations through a combination of conformance outcomes and context-based feature attribution.
- Decision-tree visualizations are written to file, for example ID-balanced-cases-random.pdf in the repository root.

## License

This repository is intended for research and experimental use.