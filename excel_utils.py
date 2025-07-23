import pandas as pd
import os
from datetime import datetime, timedelta
from collections import defaultdict
import json

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


def preprocess_budget_dataframe(dataframe):
    """
    Cleans and preprocesses the raw budget DataFrame based on specified columns.
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
        print("Warning: 'Amount' column not found after preprocessing. "
              "Ensure your file contains a column for numerical amounts.")
    
    # Ensure 'Period' is datetime (if it's a date column)
    if 'Period' in df_cleaned.columns:
        df_cleaned['Period'] = pd.to_datetime(df_cleaned['Period'], errors='coerce')
    
    return df_cleaned


def analyze_categories_dynamically(dataframe):
    """
    Dynamically analyzes all categories in the budget without hardcoding.
    
    Args:
        dataframe (pd.DataFrame): Preprocessed DataFrame
        
    Returns:
        dict: Analysis results including totals by category and type
    """
    if dataframe.empty:
        return {"error": "DataFrame is empty"}
    
    analysis = {
        "total_income": 0,
        "total_expenses": 0,
        "net_amount": 0,
        "expense_categories": {},
        "savings_categories": {},
        "category_summary": {},
        "top_expense_categories": [],
        "unique_categories": []
    }
    
    # Get all unique categories
    if 'Category' in dataframe.columns:
        analysis["unique_categories"] = dataframe['Category'].dropna().unique().tolist()
    
    # Analyze by Income/Expense type
    if 'Income/Expense' in dataframe.columns and 'Amount' in dataframe.columns:
        # Income analysis
        income_df = dataframe[dataframe['Income/Expense'].isin(['Income'])]
        analysis["total_income"] = income_df['Amount'].sum()
        
        # Expense analysis
        expense_df = dataframe[dataframe['Income/Expense'].isin(['Exp.', 'Expense'])]
        analysis["total_expenses"] = expense_df['Amount'].sum()
        
        if 'Category' in dataframe.columns:
            expense_by_category = expense_df.groupby('Category')['Amount'].sum().to_dict()
            analysis["expense_categories"] = expense_by_category
            
            # Get top expense categories
            if expense_by_category:
                sorted_expenses = sorted(expense_by_category.items(), key=lambda x: x[1], reverse=True)
                analysis["top_expense_categories"] = sorted_expenses[:10]  # Top 10
    
    # Identify savings categories (containing keywords like 'savings', 'HYSA', etc.)
    savings_keywords = ["HYSA"]
    if 'Category' in dataframe.columns:
        for category in analysis["unique_categories"]:
            if any(keyword.lower() in category.lower() for keyword in savings_keywords):
                savings_amount = dataframe[dataframe['Category'] == category]['Amount'].sum()
                analysis["savings_categories"][category] = savings_amount
    
    # Calculate net amount
    total_savings = sum(abs(amount) for amount in analysis["savings_categories"].values())

    analysis["net_amount"] = analysis["total_income"] - (analysis["total_expenses"] + total_savings)

    # Create overall category summary (all categories with their totals)
    if 'Category' in dataframe.columns and 'Amount' in dataframe.columns:
        category_totals = dataframe.groupby('Category')['Amount'].sum().to_dict()
        analysis["category_summary"] = category_totals
    
    return analysis


def get_basic_df_summary(dataframe):
    """
    Generates a comprehensive analysis of a DataFrame and returns both 
    the string summary and the raw analysis data.

    Args:
        dataframe (pd.DataFrame): The DataFrame to summarize (raw or preprocessed).
    Returns:
        tuple: (summary_string, analysis_dict) or (error_message, None)
    """
    # First, preprocess the data
    processed_df = preprocess_budget_dataframe(dataframe)

    if processed_df.empty:
        return "No data available.", None

    # Get dynamic analysis
    analysis = analyze_categories_dynamically(processed_df)
    
    if "error" in analysis:
        return f"Error in analysis: {analysis['error']}", None

    # Build comprehensive summary string
    summary_lines = [
        f"Dataset contains {len(processed_df)} transactions",
        "",
        "=== FINANCIAL OVERVIEW ===",
        f"Total Income: ${analysis['total_income']:,.2f}",
        f"Total Expenses: ${analysis['total_expenses']:,.2f}",
        f"Total Savings: ${analysis['total_savings']:,.2f}",
        f"Net Amount: ${analysis['net_amount']:,.2f}",
        ""
    ]

    # Income overview (simplified since you only have one income source)
    if analysis['total_income'] > 0:
        summary_lines.append("=== INCOME OVERVIEW ===")
        summary_lines.append(f"Total Income: ${analysis['total_income']:,.2f}")
        summary_lines.append("")

    # Top expense categories (limit to top 15 for LLM readability)
    if analysis['top_expense_categories']:
        summary_lines.append("=== TOP EXPENSE CATEGORIES ===")
        for i, (category, amount) in enumerate(analysis['top_expense_categories'][:15]):
            percentage = (amount / analysis['total_expenses'] * 100) if analysis['total_expenses'] > 0 else 0
            summary_lines.append(f"{i+1}. {category}: ${amount:,.2f} ({percentage:.1f}%)")
        summary_lines.append("")

    # Savings categories
    if analysis['savings_categories']:
        summary_lines.append("=== SAVINGS & INVESTMENTS ===")
        for category, amount in analysis['savings_categories'].items():
            summary_lines.append(f"• {category}: ${amount:,.2f}")
        summary_lines.append(f"Total Savings Activity: ${analysis['total_savings']:,.2f}")
        summary_lines.append("")

    # Show a sample of other categories for context (avoid overwhelming the LLM)
    other_categories = {k: v for k, v in analysis['category_summary'].items() 
                       if k not in analysis['expense_categories'] 
                       and k not in analysis['savings_categories']}
    
    if other_categories:
        summary_lines.append("=== OTHER CATEGORIES (Sample) ===")
        # Show top 10 by absolute value
        sorted_other = sorted(other_categories.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        for category, amount in sorted_other:
            summary_lines.append(f"• {category}: ${amount:,.2f}")
        if len(other_categories) > 10:
            summary_lines.append(f"... and {len(other_categories) - 10} more categories")
        summary_lines.append("")

    # Add total unique categories count
    summary_lines.append(f"Total Unique Categories: {len(analysis['unique_categories'])}")
    
    summary_string = "\n".join(summary_lines)
    return summary_string, analysis


def get_month_comparison_analysis(this_month_df, last_month_df, this_month_display, last_month_display):
    """
    Generates month-to-month comparison analysis and returns both string and data.
    
    Args:
        this_month_df (pd.DataFrame): Current month data
        last_month_df (pd.DataFrame): Previous month data
        this_month_display (str): Current month display name
        last_month_display (str): Previous month display name
        
    Returns:
        tuple: (comparison_string, comparison_data_dict) or (error_message, None)
    """
    this_analysis = analyze_categories_dynamically(preprocess_budget_dataframe(this_month_df))
    last_analysis = analyze_categories_dynamically(preprocess_budget_dataframe(last_month_df))
    
    if "error" in this_analysis or "error" in last_analysis:
        return "Month-to-month comparison unavailable due to data processing errors.", None
    
    # Calculate changes
    income_change = this_analysis['total_income'] - last_analysis['total_income']
    income_change_pct = (income_change / last_analysis['total_income'] * 100) if last_analysis['total_income'] != 0 else 0
    
    expense_change = this_analysis['total_expenses'] - last_analysis['total_expenses']
    expense_change_pct = (expense_change / last_analysis['total_expenses'] * 100) if last_analysis['total_expenses'] != 0 else 0
    
    savings_change = this_analysis['total_savings'] - last_analysis['total_savings']
    savings_change_pct = (savings_change / last_analysis['total_savings'] * 100) if last_analysis['total_savings'] != 0 else 0
    
    net_change = this_analysis['net_amount'] - last_analysis['net_amount']
    
    # Build comparison data dictionary
    comparison_data = {
        "this_month": {
            "display_name": this_month_display,
            "analysis": this_analysis
        },
        "last_month": {
            "display_name": last_month_display,
            "analysis": last_analysis
        },
        "changes": {
            "income": {
                "absolute": income_change,
                "percentage": income_change_pct
            },
            "expenses": {
                "absolute": expense_change,
                "percentage": expense_change_pct
            },
            "savings": {
                "absolute": savings_change,
                "percentage": savings_change_pct
            },
            "net_amount": {
                "absolute": net_change
            }
        },
        "category_changes": []
    }
    
    comparison_lines = [
        f"=== MONTH-TO-MONTH COMPARISON: {last_month_display} vs {this_month_display} ===",
        "",
        "KEY CHANGES:",
        f"• Income: ${last_analysis['total_income']:,.2f} → ${this_analysis['total_income']:,.2f} "
        f"({income_change:+,.2f}, {income_change_pct:+.1f}%)",
        f"• Expenses: ${last_analysis['total_expenses']:,.2f} → ${this_analysis['total_expenses']:,.2f} "
        f"({expense_change:+,.2f}, {expense_change_pct:+.1f}%)",
        f"• Savings: ${last_analysis['total_savings']:,.2f} → ${this_analysis['total_savings']:,.2f} "
        f"({savings_change:+,.2f}, {savings_change_pct:+.1f}%)",
        f"• Net Amount: ${last_analysis['net_amount']:,.2f} → ${this_analysis['net_amount']:,.2f} "
        f"({net_change:+,.2f})",
        "",
        "TOP CATEGORY CHANGES:"
    ]
    
    # Compare categories
    all_categories = set(this_analysis['category_summary'].keys()) | set(last_analysis['category_summary'].keys())
    category_changes = []
    
    for category in all_categories:
        this_amount = this_analysis['category_summary'].get(category, 0)
        last_amount = last_analysis['category_summary'].get(category, 0)
        change = this_amount - last_amount
        
        if abs(change) > 1:  # Only show changes > $1
            change_pct = (change / abs(last_amount) * 100) if last_amount != 0 else float('inf')
            category_change_data = {
                "category": category,
                "last_amount": last_amount,
                "this_amount": this_amount,
                "change": change,
                "change_percentage": change_pct
            }
            category_changes.append(category_change_data)
    
    # Sort by absolute change and show top 10
    category_changes.sort(key=lambda x: abs(x['change']), reverse=True)
    comparison_data["category_changes"] = category_changes
    
    for change_data in category_changes[:10]:
        category = change_data["category"]
        last_amt = change_data["last_amount"]
        this_amt = change_data["this_amount"]
        change = change_data["change"]
        change_pct = change_data["change_percentage"]
        
        if change_pct == float('inf'):
            comparison_lines.append(f"• {category}: ${last_amt:,.2f} → ${this_amt:,.2f} (NEW CATEGORY)")
        else:
            comparison_lines.append(f"• {category}: ${last_amt:,.2f} → ${this_amt:,.2f} "
                                   f"({change:+,.2f}, {change_pct:+.1f}%)")
    
    comparison_string = "\n".join(comparison_lines)
    return comparison_string, comparison_data


def run_complete_analysis():
    """
    Runs complete analysis and returns all results for use in main.py
    
    Returns:
        dict: Complete analysis results including:
            - this_month_summary: string
            - this_month_analysis: dict
            - last_month_summary: string  
            - last_month_analysis: dict
            - comparison_summary: string
            - comparison_data: dict
            - file_info: dict with paths and display names
    """
    # Get file information
    this_month_full_path, this_month_display, last_month_full_path, last_month_display = get_last_month_this_month_info()
    
    # Load data
    raw_df_this_month = load_excel_file(this_month_full_path)
    raw_df_last_month = load_excel_file(last_month_full_path)
    
    results = {
        "file_info": {
            "this_month_path": this_month_full_path,
            "this_month_display": this_month_display,
            "last_month_path": last_month_full_path,
            "last_month_display": last_month_display
        },
        "this_month_summary": None,
        "this_month_analysis": None,
        "last_month_summary": None,
        "last_month_analysis": None,
        "comparison_summary": None,
        "comparison_data": None,
        "errors": []
    }
    
    # Analyze this month
    if not raw_df_this_month.empty:
        this_month_summary, this_month_analysis = get_basic_df_summary(raw_df_this_month)
        results["this_month_summary"] = this_month_summary
        results["this_month_analysis"] = this_month_analysis
    else:
        results["errors"].append(f"Could not load {this_month_display} data")
        results["this_month_summary"] = f"No data available for {this_month_display}"
    
    # Analyze last month
    if not raw_df_last_month.empty:
        last_month_summary, last_month_analysis = get_basic_df_summary(raw_df_last_month)
        results["last_month_summary"] = last_month_summary
        results["last_month_analysis"] = last_month_analysis
    else:
        results["errors"].append(f"Could not load {last_month_display} data")
        results["last_month_summary"] = f"No data available for {last_month_display}"
    
    # Generate comparison if both months available
    if not raw_df_this_month.empty and not raw_df_last_month.empty:
        comparison_summary, comparison_data = get_month_comparison_analysis(
            raw_df_this_month, raw_df_last_month, this_month_display, last_month_display
        )
        results["comparison_summary"] = comparison_summary
        results["comparison_data"] = comparison_data
    else:
        results["comparison_summary"] = "Month-to-month comparison unavailable - missing data for one or both months"
    
    return results


# Update the analyze_categories_dynamically function to include total_savings
def analyze_categories_dynamically(dataframe):
    """
    Dynamically analyzes all categories in the budget without hardcoding.
    
    Args:
        dataframe (pd.DataFrame): Preprocessed DataFrame
        
    Returns:
        dict: Analysis results including totals by category and type
    """
    if dataframe.empty:
        return {"error": "DataFrame is empty"}
    
    analysis = {
        "total_income": 0,
        "total_expenses": 0,
        "total_savings": 0,  # Added this
        "net_amount": 0,
        "expense_categories": {},
        "savings_categories": {},
        "category_summary": {},
        "top_expense_categories": [],
        "unique_categories": []
    }
    
    # Get all unique categories
    if 'Category' in dataframe.columns:
        analysis["unique_categories"] = dataframe['Category'].dropna().unique().tolist()
    
    # Analyze by Income/Expense type
    if 'Income/Expense' in dataframe.columns and 'Amount' in dataframe.columns:
        # Income analysis
        income_df = dataframe[dataframe['Income/Expense'].isin(['Income'])]
        analysis["total_income"] = income_df['Amount'].sum()
        
        # Expense analysis
        expense_df = dataframe[dataframe['Income/Expense'].isin(['Exp.', 'Expense'])]
        analysis["total_expenses"] = expense_df['Amount'].sum()
        
        if 'Category' in dataframe.columns:
            expense_by_category = expense_df.groupby('Category')['Amount'].sum().to_dict()
            analysis["expense_categories"] = expense_by_category
            
            # Get top expense categories
            if expense_by_category:
                sorted_expenses = sorted(expense_by_category.items(), key=lambda x: x[1], reverse=True)
                analysis["top_expense_categories"] = sorted_expenses[:10]  # Top 10
    
    # Identify savings categories (containing keywords like 'savings', 'HYSA', etc.)
    savings_keywords = ['savings', 'hysa', 'save', 'investment', 'deposit']
    if 'Category' in dataframe.columns:
        for category in analysis["unique_categories"]:
            if any(keyword.lower() in category.lower() for keyword in savings_keywords):
                savings_amount = dataframe[dataframe['Category'] == category]['Amount'].sum()
                analysis["savings_categories"][category] = abs(savings_amount)  # Use abs for consistent positive values
    
    # Calculate total savings
    analysis["total_savings"] = sum(analysis["savings_categories"].values())
    
    # Calculate net amount (income - expenses - savings)
    analysis["net_amount"] = analysis["total_income"] - analysis["total_expenses"] - analysis["total_savings"]
    
    # Create overall category summary (all categories with their totals)
    if 'Category' in dataframe.columns and 'Amount' in dataframe.columns:
        category_totals = dataframe.groupby('Category')['Amount'].sum().to_dict()
        analysis["category_summary"] = category_totals
    
    return analysis


def get_month_comparison_analysis(this_month_df, last_month_df, this_month_display, last_month_display):
    """
    Generates month-to-month comparison analysis for inclusion in LLM prompt.
    This is an additional function that can be called from main.py if needed.
    
    Args:
        this_month_df (pd.DataFrame): Current month data
        last_month_df (pd.DataFrame): Previous month data
        this_month_display (str): Current month display name
        last_month_display (str): Previous month display name
        
    Returns:
        str: Comparison analysis string
    """
    this_analysis = analyze_categories_dynamically(preprocess_budget_dataframe(this_month_df))
    last_analysis = analyze_categories_dynamically(preprocess_budget_dataframe(last_month_df))
    
    if "error" in this_analysis or "error" in last_analysis:
        return "Month-to-month comparison unavailable due to data processing errors."
    
    comparison_lines = [
        f"=== MONTH-TO-MONTH COMPARISON: {last_month_display} vs {this_month_display} ===",
        "",
        "KEY CHANGES:",
    ]
    
    # Income comparison
    income_change = this_analysis['total_income'] - last_analysis['total_income']
    income_change_pct = (income_change / last_analysis['total_income'] * 100) if last_analysis['total_income'] != 0 else 0
    comparison_lines.append(f"• Income: ${last_analysis['total_income']:,.2f} → ${this_analysis['total_income']:,.2f} "
                           f"({income_change:+,.2f}, {income_change_pct:+.1f}%)")
    
    # Expense comparison
    expense_change = this_analysis['total_expenses'] - last_analysis['total_expenses']
    expense_change_pct = (expense_change / last_analysis['total_expenses'] * 100) if last_analysis['total_expenses'] != 0 else 0
    comparison_lines.append(f"• Expenses: ${last_analysis['total_expenses']:,.2f} → ${this_analysis['total_expenses']:,.2f} "
                           f"({expense_change:+,.2f}, {expense_change_pct:+.1f}%)")
    
    # Net amount comparison
    net_change = this_analysis['net_amount'] - last_analysis['net_amount']
    comparison_lines.append(f"• Net Amount: ${last_analysis['net_amount']:,.2f} → ${this_analysis['net_amount']:,.2f} "
                           f"({net_change:+,.2f})")
    
    comparison_lines.append("")
    comparison_lines.append("TOP CATEGORY CHANGES:")
    
    # Compare categories
    all_categories = set(this_analysis['category_summary'].keys()) | set(last_analysis['category_summary'].keys())
    category_changes = []
    
    for category in all_categories:
        this_amount = this_analysis['category_summary'].get(category, 0)
        last_amount = last_analysis['category_summary'].get(category, 0)
        change = this_amount - last_amount
        
        if abs(change) > 1:  # Only show changes > $1
            change_pct = (change / abs(last_amount) * 100) if last_amount != 0 else float('inf')
            category_changes.append((category, last_amount, this_amount, change, change_pct))
    
    # Sort by absolute change and show top 10
    category_changes.sort(key=lambda x: abs(x[3]), reverse=True)
    for category, last_amt, this_amt, change, change_pct in category_changes[:10]:
        if change_pct == float('inf'):
            comparison_lines.append(f"• {category}: ${last_amt:,.2f} → ${this_amt:,.2f} (NEW CATEGORY)")
        else:
            comparison_lines.append(f"• {category}: ${last_amt:,.2f} → ${this_amt:,.2f} "
                                   f"({change:+,.2f}, {change_pct:+.1f}%)")
    
    return "\n".join(comparison_lines)


def compare_months(this_month_df, last_month_df, this_month_display, last_month_display):
    """
    Compares two months of budget data and generates insights for LLM analysis.
    
    Args:
        this_month_df (pd.DataFrame): Current month data
        last_month_df (pd.DataFrame): Previous month data
        this_month_display (str): Current month display name
        last_month_display (str): Previous month display name
        
    Returns:
        str: Comparison summary for LLM
    """
    this_analysis = analyze_categories_dynamically(preprocess_budget_dataframe(this_month_df))
    last_analysis = analyze_categories_dynamically(preprocess_budget_dataframe(last_month_df))
    
    if "error" in this_analysis or "error" in last_analysis:
        return "Error: Could not analyze one or both months for comparison."
    
    comparison_lines = [
        f"=== MONTH-TO-MONTH COMPARISON: {last_month_display} vs {this_month_display} ===",
        "",
        "FINANCIAL OVERVIEW CHANGES:",
    ]
    
    # Income comparison
    income_change = this_analysis['total_income'] - last_analysis['total_income']
    income_change_pct = (income_change / last_analysis['total_income'] * 100) if last_analysis['total_income'] != 0 else 0
    comparison_lines.append(f"• Income: ${last_analysis['total_income']:,.2f} → ${this_analysis['total_income']:,.2f} "
                           f"({income_change:+,.2f}, {income_change_pct:+.1f}%)")
    
    # Expense comparison
    expense_change = this_analysis['total_expenses'] - last_analysis['total_expenses']
    expense_change_pct = (expense_change / last_analysis['total_expenses'] * 100) if last_analysis['total_expenses'] != 0 else 0
    comparison_lines.append(f"• Expenses: ${last_analysis['total_expenses']:,.2f} → ${this_analysis['total_expenses']:,.2f} "
                           f"({expense_change:+,.2f}, {expense_change_pct:+.1f}%)")
    
    # Net amount comparison
    net_change = this_analysis['net_amount'] - last_analysis['net_amount']
    comparison_lines.append(f"• Net Amount: ${last_analysis['net_amount']:,.2f} → ${this_analysis['net_amount']:,.2f} "
                           f"({net_change:+,.2f})")
    
    comparison_lines.append("")
    comparison_lines.append("CATEGORY CHANGES (Top 10 by absolute change):")
    
    # Compare categories
    all_categories = set(this_analysis['category_summary'].keys()) | set(last_analysis['category_summary'].keys())
    category_changes = []
    
    for category in all_categories:
        this_amount = this_analysis['category_summary'].get(category, 0)
        last_amount = last_analysis['category_summary'].get(category, 0)
        change = this_amount - last_amount
        
        if change != 0:
            change_pct = (change / abs(last_amount) * 100) if last_amount != 0 else float('inf')
            category_changes.append((category, last_amount, this_amount, change, change_pct))
    
    # Sort by absolute change and show top 10
    category_changes.sort(key=lambda x: abs(x[3]), reverse=True)
    for category, last_amt, this_amt, change, change_pct in category_changes[:10]:
        if change_pct == float('inf'):
            comparison_lines.append(f"• {category}: ${last_amt:,.2f} → ${this_amt:,.2f} (NEW)")
        else:
            comparison_lines.append(f"• {category}: ${last_amt:,.2f} → ${this_amt:,.2f} "
                                   f"({change:+,.2f}, {change_pct:+.1f}%)")
    
    return "\n".join(comparison_lines)


def get_last_month_this_month_info():
    """Get file paths and display names for current and previous month"""
    today = datetime.now()
    current_year = today.year
    budget_base_folder = os.path.join(os.getcwd(), str(current_year))

    this_month_name = today.strftime('%B').capitalize()
    this_month_filename = f"{this_month_name}_{current_year}.xlsx"
    this_month_full_path = os.path.join(budget_base_folder, this_month_filename)
    this_month_display = today.strftime('%B %Y')

    last_month_date = today.replace(day=1) - timedelta(days=1)
    last_month_name = last_month_date.strftime('%B').capitalize()
    last_month_year = last_month_date.year
    last_month_filename = f"{last_month_name}_{last_month_year}.xlsx"
    last_month_full_path = os.path.join(budget_base_folder, last_month_filename)
    last_month_display = last_month_date.strftime('%B %Y')
    
    return this_month_full_path, this_month_display, last_month_full_path, last_month_display


def generate_llm_prompt(this_month_summary, last_month_summary, comparison_summary):
    """
    Generate a comprehensive prompt for LLM analysis
    
    Args:
        this_month_summary (str): Current month analysis
        last_month_summary (str): Previous month analysis
        comparison_summary (str): Month-to-month comparison
        
    Returns:
        str: Formatted prompt for LLM
    """
    prompt = f"""
Please analyze the following budget data and provide insights:

{comparison_summary}

DETAILED CURRENT MONTH DATA:
{this_month_summary}

DETAILED PREVIOUS MONTH DATA:
{last_month_summary}

Please provide:
1. Key financial insights and trends
2. Areas of concern or improvement opportunities
3. Spending pattern analysis
4. Recommendations for better budget management
5. Notable changes between months and their potential causes

Focus on actionable insights rather than just restating the numbers.
"""
    return prompt


# --- Test functionality when excel_utils.py is run directly ---
if __name__ == "__main__":
    print("--- Testing Enhanced Budget Analysis Script ---")
    this_month_full_path, this_month_display, last_month_full_path, last_month_display = get_last_month_this_month_info()

    print(f"\nTargeting this month's file: '{this_month_full_path}'")
    print(f"Targeting last month's file: '{last_month_full_path}'")

    # Load data
    print("\n--- Loading Data ---")
    raw_df_this_month = load_excel_file(this_month_full_path)
    raw_df_last_month = load_excel_file(last_month_full_path)

    if not raw_df_this_month.empty:
        print(f"\nLoaded '{this_month_display}' data. Shape: {raw_df_this_month.shape}")
        print("Columns:", raw_df_this_month.columns.tolist())
    else:
        print(f"\nCould not load '{this_month_display}' data.")

    if not raw_df_last_month.empty:
        print(f"\nLoaded '{last_month_display}' data. Shape: {raw_df_last_month.shape}")
        print("Columns:", raw_df_last_month.columns.tolist())
    else:
        print(f"\nCould not load '{last_month_display}' data.")

    # Generate summaries for testing
    if not raw_df_this_month.empty:
        this_month_summary = get_basic_df_summary(raw_df_this_month)
        print(f"\n=== {this_month_display} Summary (for LLM) ===")
        print(this_month_summary)
    else:
        this_month_summary = f"No data available for {this_month_display}"

    if not raw_df_last_month.empty:
        last_month_summary = get_basic_df_summary(raw_df_last_month)
        print(f"\n=== {last_month_display} Summary (for LLM) ===")
        print(last_month_summary)
    else:
        last_month_summary = f"No data available for {last_month_display}"

    # Generate comparison (optional - can be used in prompts if needed)
    if not raw_df_this_month.empty and not raw_df_last_month.empty:
        comparison_summary = get_month_comparison_analysis(raw_df_this_month, raw_df_last_month, 
                                          this_month_display, last_month_display)
        print(f"\n=== Month Comparison (Optional for Prompts) ===")
        print(comparison_summary)

    print("\n--- Analysis Complete ---")