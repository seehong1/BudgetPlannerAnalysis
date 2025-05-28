import os
from dotenv import load_dotenv

import excel_utils
import gemini_utils
import config

def run_llm_analysis_tests():
    print("--- Starting LLM Analysis Tests ---")

    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not gemini_api_key:
        print("ERROR: GEMINI_API_KEY not found in .env file. Please set it up.")
        return

    if not gemini_utils.configure_gemini_api(gemini_api_key):
        print("Failed to configure Gemini API. Aborting tests.")
        return

    llm_model_name = config.GEMINI_MODEL

    this_month_full_path, this_month_display, last_month_full_path, last_month_display = excel_utils.get_last_month_this_month_info()
    print(f"\nAttempting to load data for '{this_month_display}' and '{last_month_display}'...")

    # Load and Get Summaries of DataFrames
    current_month_df = excel_utils.load_excel_file(this_month_full_path)
    last_month_df = excel_utils.load_excel_file(last_month_full_path)

    if current_month_df.empty or last_month_df.empty:
        print("ERROR: Could not load both current and last month's budget planner files. Please ensure they exist.")
        print(f"Current Month File Expected: {this_month_full_path}")
        print(f"Last Month File Expected: {last_month_full_path}")
        return

    current_month_summary = excel_utils.get_basic_df_summary(current_month_df)
    last_month_summary = excel_utils.get_basic_df_summary(last_month_df)

    print("\n--- Sending Prompts to Gemini for Analysis ---")

    # Prompt 1: Month-over-Month Comparison

    prompt_comparison = f"""
        Please analyze and compare the household budget planner data from two consecutive months:

        **{this_month_display} Budget Planner Summary:**
        {current_month_df}
        Total income and expense: {current_month_summary}
    
        **{last_month_display} Budget Planner Summary:**
        {last_month_df}
        "Total income and expense: {last_month_summary}

        Provide a concise comparison of:
        - Overall spending trends.
        - Suggest areas for improvement or interesting insights.
        """
    
    if config.LANGUAGE == "Korean" or config.LANGUAGE == "한국어":
        prompt_comparison += "\n 답변을 한국어로 해줘"

    
    print("\n[Prompt 1: Month-over-Month Comparison]")
    print("Getting response from Gemini (this may take a moment)...")
    response_comparison = gemini_utils.get_gemini_response(prompt_comparison, model_name=llm_model_name)
    if response_comparison:
        print("\n--- Gemini's Comparison Analysis ---")
        print(response_comparison)
    else:
        print("\nFailed to get comparison analysis from Gemini.")

    # Prompt 2: Detailed Expense Breakdown for Current Month
    prompt_current_expenses = f"""
    Here is the household budget planner data for this month: {current_month_df}:

    Based on this data, please:
    - Sum up the totals in the main categories: Groceries, Restaruants, Transportations, Social Life, Rent, Utilities from the full data provided
    - List the top 3-5 highest expense categories and their approximate amounts.
    - Provide a brief summary of overall spending habits for this month.
    """

    if config.LANGUAGE == "Korean" or config.LANGUAGE == "한국어":
        prompt_current_expenses += "\n 답변을 한국어로 해줘"

    print("\n[Prompt 2: Detailed Expense Breakdown for Current Month]")
    print("Getting response from Gemini (this may take a moment)...")
    response_current_expenses = gemini_utils.get_gemini_response(prompt_current_expenses, model_name=llm_model_name)
    if response_current_expenses:
        print("\n--- Gemini's Current Month Expense Analysis ---")
        print(response_current_expenses)
    else:
        print("\nFailed to get current month expense analysis from Gemini.")

    print("\n--- LLM Analysis Tests Finished ---")

if __name__ == "__main__":
    run_llm_analysis_tests()