# Context-Enriched Declarative Constraints

This repository implements a Python workflow for analyzing declarative process constraints on BPIC19 purchase-to-pay event data. It supports:

- interactive constraint selection,
- conformance evaluation for unary and binary constraints,
- context-based feature attribution with conditional Shapley values,
- decision-tree-based explanations of constraint satisfaction and violation,
- integrated analysis across multiple constraints, and
- export of decision-tree visualizations and analysis results.

## What the project does

- Loads event-log data and declarative constraints from XES/JSON inputs.
- Evaluates unary and binary constraints such as AtMostOne, End, and AlternatePrecedence.
- Computes context-based feature importance using conditional Shapley values.
- Builds decision trees to explain whether cases satisfy or violate a selected constraint.
- Extracts activation conditions from the decision tree as human-readable rules.
- Supports an integrated analysis mode for evaluating multiple promising constraints together.
- Saves generated outputs such as decision-tree PDFs and CSV result files.

## Repository structure

- [driver.py](driver.py) – Entry point. Loads the data, lists the available constraints, and prompts the user to choose an analysis mode.
- [load_data.py](load_data.py) – Loads event logs and declarative constraints from the repository inputs.
- [drop_irrelevant_attributes.py](drop_irrelevant_attributes.py) – Drop irrelevant attributes of events for the analysis.
- [check_conformance.py](check_conformance.py) – Evaluates the selected constraint(s) and prepares the analysis dataset.
- [find_shapley_values.py](find_shapley_values.py) – Computes conditional Shapley values for the selected features.
- [build_decision_trees.py](build_decision_trees.py) – Builds decision trees, extracts activation conditions, and exports visualizations.
- [declarative_constraints.json](declarative_constraints.json) – Example declarative constraints used by the workflow.
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

After startup, the program offers three options:

- Analyze a single selected constraint by ID.
- Run an integrated analysis over multiple constraints.
- Exit the program.

## Notes

- The workflow is interactive and prompts for the analysis mode after loading the data.
- Decision-tree visualizations are exported as PDF files, for example C2-decision-tree.pdf.
- Integrated analysis results are saved as CSV files in the repository root.
- For PDF rendering, Graphviz must be installed and available on your system PATH.

## License

This repository is intended for research and experimental use.