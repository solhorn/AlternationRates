# Alternation Analysis for Y-/T-Maze DNMP Experiments

Python script for analyzing success rates in Y- and T-maze delayed non-match-to-position (DNMP) experiments across postnatal development.

## Requirements

Install the required Python packages before running the script:

```bash
pip install pandas numpy matplotlib scipy statsmodels openpyxl
```

## Input Files

The script expects Excel files containing behavioral and animal metadata.

Examples from the current script:

* `alternation_with_time_all.xlsx`
  Main dataset containing values such as ID, age, daily success rate, sex, and experiment type

* `info y maze wo empty.xlsx`
  Trial-level dataset used for significance threshold calculations

* `info rats.xlsx`
  Metadata about animals, such as sex and ID

## Expected Columns

Your input data should contain columns such as:

* `ID`
* `age`
* `sex`
* `success_rate`
* `experiment_type`

For some functions, the script also expects:

* `type of experiment`

Make sure column names in the Excel files match the names used in the script.

## Usage

1. Update the file paths in the script so they point to your local Excel files and output folder.
2. Choose the experiment type(s) you want to analyze.
3. Run the script in Spyder, VS Code, Jupyter, or from the command line.

Example:

```python
experiment_types = ['sd']
```
Then run the script.


## Output

The script generates plots such as:

* individual success rate across age
* mean success rate by sex
* mean and median success rate across age
* regression plots for success rate versus age
* comparisons between experiment types

Daily significance thresholds are calculated using binomial test, and presented on plots.

## Statistical Methods

Depending on the analysis, the script uses:

* binomial tests for significance threshold calculation
* Shapiro–Wilk test for normality
* Welch’s t-test for group comparisons when data are normally distributed
* Mann–Whitney U test when normality assumptions are not met
* linear regression using ordinary least squares (OLS)
* Pearson and Spearman correlations

## Example Workflow

A typical workflow in this script is:

1. Load behavioral data from Excel
2. Filter by experiment type
3. Compute daily significance thresholds
4. Generate plots for individuals, sexes, and group means
5. Run regression analyses
6. Compare experiment types across selected ages

## Notes

* The script currently uses absolute local file paths. These should be changed before sharing the code with others.
* The script is designed for behavioral neuroscience datasets and may need adaptation for other formats.

## Author

Solveig Horn
solhorn4@gmail.com
