import pandas as pd
import os
from datetime import datetime, timedelta

def load_excel_file(file_full_path):
    """
    Loads an Excel file into a pandas DataFrame.
    Handles FileNotFoundError and general Excel reading errors.

    Args:
        file_full_path (str): The absolute path to the Excel file (.xlsx).

    Returns:
        pandas.DataFrame: The loaded DataFrame, or an empty DataFrame if an error occurs.
    """
    if not os.path.exists(file_full_path):
        print(f"Error: File not found at '{file_full_path}'")
        return pd.DataFrame() # Return empty DataFrame
    
    try:
        df = pd.read_excel(file_full_path)
        return df
    except Exception as e:
        print(f"Error reading Excel file '{file_full_path}': {e}")
        return pd.DataFrame()


def preprocess_ledger_dataframe(dataframe):
    """
    Cleans and preprocesses the raw ledger DataFrame based on specified columns.
    This includes dropping unnecessary columns and renaming others.

    Args:
        dataframe (pd.DataFrame): The raw DataFrame loaded from the Excel file.

    Returns:
        pd.DataFrame: The cleaned and preprocessed DataFrame.
    """
    if dataframe.empty:
        print("Input DataFrame is empty. No preprocessing performed.")
        return dataframe # Return empty if input is empty

    df_cleaned = dataframe.copy()

    # Delete the last column first
    if not df_cleaned.empty and len(df_cleaned.columns) > 0:
        last_column_name = df_cleaned.columns[-1] # Get the name of the last column
        df_cleaned = df_cleaned.drop(columns=last_column_name, errors='ignore')
    else:
        print("Warning: Cannot determine last column to drop (DataFrame is empty or has no columns).")

    # next delete the column USD
    df_cleaned = df_cleaned.drop(columns=['USD'], errors='ignore')

    # Ensure 'Amount' is numeric. Convert non-numeric to NaN, then fill with 0.
    if 'Amount' in df_cleaned.columns:
        df_cleaned['Amount'] = pd.to_numeric(df_cleaned['Amount'], errors='coerce').fillna(0)
    else:
        print("Warning: 'Amounts' column not found after preprocessing. "
              "Ensure your file contains a column for numerical amounts.")
    
    # Ensure 'Period' is datetime (if it's a date column)
    if 'Period' in df_cleaned.columns:
        df_cleaned['Period'] = pd.to_datetime(df_cleaned['Period'], errors='coerce')
    return df_cleaned

def get_basic_df_summary(dataframe):
    """
    Generates a basic string summary of a DataFrame's head for LLM input.
    This function first preprocesses the DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame to summarize (raw or preprocessed).
    Returns:
        str: A formatted string representing the DataFrame's head, or "No data available."
    """

    #TODO: To be expanded
    summary_str = ""

    # First, preprocess the data
    processed_df = preprocess_ledger_dataframe(dataframe)

    if processed_df.empty:
        return "No data available in this ledger after preprocessing."

    # Processing total income and total expense
    income_df = processed_df[processed_df['Income/Expense'].isin(['Income'])]
    total_income = income_df['Amount'].sum()

    expense_df = processed_df[processed_df['Income/Expense'].isin(['Exp.'])]
    total_expense = expense_df['Amount'].sum()

    summary_str += f"\nTotal income: {total_income}"
    summary_str += f"\nTotal expense: {total_expense}"
    
    return summary_str

def get_last_month_this_month_info():
    today = datetime.now()
    current_year = today.year
    ledger_base_folder = os.path.join(os.getcwd(), str(current_year))

    this_month_name = today.strftime('%B').capitalize()
    this_month_filename = f"{this_month_name}_{current_year}.xlsx"
    this_month_full_path = os.path.join(ledger_base_folder, this_month_filename)
    this_month_display = today.strftime('%B %Y')

    last_month_date = today.replace(day=1) - timedelta(days=1)
    last_month_name = last_month_date.strftime('%B').capitalize()
    last_month_year = last_month_date.year
    last_month_filename = f"{last_month_name}_{last_month_year}.xlsx"
    last_month_full_path = os.path.join(ledger_base_folder, last_month_filename)
    last_month_display = last_month_date.strftime('%B %Y')
    return this_month_full_path, this_month_display, last_month_full_path, last_month_display


# --- Test functionality when excel_utils.py is run directly ---
if __name__ == "__main__":
    print("--- Testing excel_utils.py functions (Excel only) ---")
    this_month_full_path, this_month_display, last_month_full_path, last_month_display = get_last_month_this_month_info()

    print(f"\nTargeting this month's file: '{this_month_full_path}'")
    print(f"Targeting last month's file: '{last_month_full_path}'")

    # --- Test 1: Loading current and last month's files ---
    print("\n--- Test 1: Loading monthly ledger files ---")
    
    raw_df_this_month = load_excel_file(this_month_full_path)
    raw_df_last_month = load_excel_file(last_month_full_path)

    if not raw_df_this_month.empty:
        print(f"\nLoaded '{this_month_display}' RAW data. Shape: {raw_df_this_month.shape}")
        print("RAW columns:", raw_df_this_month.columns.tolist())
    else:
        print(f"\nCould not load '{this_month_display}' RAW data.")

    if not raw_df_last_month.empty:
        print(f"\nLoaded '{last_month_display}' RAW data. Shape: {raw_df_last_month.shape}")
        print("RAW columns:", raw_df_last_month.columns.tolist())
    else:
        print(f"\nCould not load '{last_month_display}' RAW data.")
        
    # --- Test 2: Preprocessing and getting basic summary ---
    print("\n--- Test 2: Preprocessing and getting basic summary (for LLM) ---")
    
    if not raw_df_this_month.empty:
        print(f"\n--- Preprocessing and Summarizing {this_month_display} ---")
        # Call get_basic_df_summary, which internally calls preprocess_ledger_dataframe
        summary_this = get_basic_df_summary(raw_df_this_month) 
        print("\nSummary for LLM (first 5 rows of processed data):")
        print(summary_this)
    else:
        print(f"\nSkipping preprocessing and summary for {this_month_display} due to loading failure.")

    if not raw_df_last_month.empty:
        print(f"\n--- Preprocessing and Summarizing {last_month_display} ---")
        # Call get_basic_df_summary, which internally calls preprocess_ledger_dataframe
        summary_last = get_basic_df_summary(raw_df_last_month) 
        print("\nSummary for LLM (first 5 rows of processed data):")
        print(summary_last)
    else:
        print(f"\nSkipping preprocessing and summary for {last_month_display} due to loading failure.")
        
    print("\n--- Finished excel_utils.py tests ---")