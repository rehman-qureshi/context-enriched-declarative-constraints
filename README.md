# Context-Enriched Declarative Constraints

This repository contains a small Python workflow for analyzing declarative process constraints on the BPIC19 purchase-to-pay event log. The project combines constraint conformance checking with explainability methods based on conditional Shapley values and decision trees.

## Overview

The codebase is designed to:

- load event-log data and declarative constraints,
- evaluate selected unary and binary constraints such as AtMostOne, End, and AlternatePrecedence,
- compute feature importance using conditional Shapley values with binning,
- build decision trees to explain why cases are classified as satisfied or violated,
- export the resulting tree visualizations as SVG, PNG, and PDF files.

## Repository Structure

- `driver.py` – command-line entry point that loads the data, shows available constraints, and lets the user select one for analysis.
- `load_data.py` – loads event logs and declarative constraint definitions from disk.
- `check_conformance.py` – evaluates the selected constraint for each case and prepares the data for explanation.
- `find_shapley_values.py` – computes conditional Shapley values for the selected features.
- `build_decision_trees.py` – trains a decision tree and exports a visualization.
- `declarative_constraints.json` – example declarative constraints used by the analysis.
- `BPIC19_3way_IbeforeGR_standardPO_complete.xes` and `bpic19_invoice_before_gr_all_cases.csv` – example event-log data files.

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

The visualization code also relies on `matplotlib` and `graphviz`, so these may need to be installed separately if they are not already available in your environment.

## Usage

Run the main script as follows:

```bash
python driver.py <events_logs_path> <declarative_constraints_path>
```

Example:

```bash
python driver.py BPIC19_3way_IbeforeGR_standardPO_complete.xes declarative_constraints.json
```

After startup, the program prints the available constraints and prompts you to enter the ID of the constraint you want to analyze.

## Current Implementation Notes

- The loader currently reads the preprocessed CSV file `bpic19_invoice_before_gr_all_cases.csv` for efficiency, rather than directly using the XES file path passed to the driver.
- The analysis workflow focuses on explaining constraint violations by combining conformance outcomes with context-based feature attribution.
- Decision tree visualizations are exported as `try-balanced-cases-random.svg`, `.png`, and `.pdf`.

## Example Constraint Types

The sample JSON file includes examples of:

- `AtMostOne`
- `End`
- `AlternatePrecedence`

These constraints are evaluated over cases from the BPIC19 process log.

## License

This repository is intended for research and experimental use.