# pnc_virtual_wallet_parse
Script to parse the monthly statement pdfs for a Virtual Wallet Spend account. May help with tax prep.

For traditional accounts, aggregate monthly csv exports from PNC into a filterable spreadsheet.
For virtual wallet accounts, parse monthly statements (in pdf form) and then aggregate into a filterable spreadsheet.

## Setup

### Versions

- Python 2.7.13
- Anaconda - conda 4.3.29
- MacOSX - 10.12.6

### Required modules

- pandas, numpy, scipy, textract, os, re, unicodedata, string, datetime

## Usage

1. Manually export the appropriate statements (CSV or PDF) from PNC.com

2. Run the following functions as appropriate (in something like iPython)

```
year_to_analyze = 2017
combine_monthly_statements_for_year('./data/transactions 2017 acct yyyy', year_to_analyze, './data/acctyyyy_2017.csv')
parse_pnc_statement_pdf('./data/spend account', year_to_analyze, './data/spend_acctzzzz_2017.csv')
```

```
def combine_monthly_statements_for_year(folder_path, year_to_analyze, save_to_fpath):
	"""
	For traditional accounts, aggregate monthly csv exports from PNC into a filterable spreadsheet.
	As of March 2018, PNC allows month-by-month monthly export of transaction activity in traditional accounts

	Aggregate from a folder of monthly activity csv exports from PNC.com for a given year (i.e. 12 files, if the account was open all year)
	
	combine_monthly_statements_for_year(folder_path, year_to_analyze, save_to_fpath)
	"""
```

```
def parse_pnc_statement_pdf(folder_path, year_to_analyze, save_to_fpath):
	""" For virtual wallet accounts, parse monthly statements (in pdf form) and then aggregate into a filterable spreadsheet. 
	As of March 2018, PNC.com has no means of exporting csv's for transaction activity in virtual accounts.

	Aggregate a folder of pdf monthly statements from PNC.com for input into a spreadsheet for easy filtering and categorization.

	parse_pnc_statement_pdf(folder_path, year_to_analyze, save_to_fpath) 
	"""
```

3. Import the csv that the program outputs into a spreadsheet program of your choice. Do the appropriate filtering, labeling for tax purposes or otherwise, and potentially, create a pivot table to aggregate the results.