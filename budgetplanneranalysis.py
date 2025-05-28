import os
import markdown
from dotenv import load_dotenv

import config
import excel_utils
import gemini_utils
import email_utils

def run_money_manager_analysis():
    print("--- Starting Budget Planner Analysis Script ---")

    # --- 1. Configuration and API Key Setup ---
    load_dotenv() # Load .env file for the main script's execution

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    EMAIL_APP_PASSWORD = os.getenv("SENDER_PASSWORD") # Using SENDER_PASSWORD from .env
    
    # Basic validation for essential credentials
    if GEMINI_API_KEY is None:
        raise ValueError("GEMINI_API_KEY environment variable is not set in your .env file.")
    if EMAIL_APP_PASSWORD is None:
        if config.EMAIL_ENABLED: # Only raise error if email is enabled and password is missing
            raise ValueError("SENDER_PASSWORD (Email App Password) environment variable is not set in your .env file, but EMAIL_SEND is True.")
        else:
            print("Password is not required - ENAIL_SEND = False")

    # Configure Gemini API (this needs to happen once)
    gemini_utils.configure_gemini_api(GEMINI_API_KEY)

    # Dynamically Determine File Paths and Display Names ---
    this_month_full_path, this_month_display, last_month_full_path, last_month_display = excel_utils.get_last_month_this_month_info()

    try:
        df_this_month = excel_utils.load_excel_file(this_month_full_path)
        df_last_month = excel_utils.load_excel_file(last_month_full_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure your Excel files are correctly named (e.g., 'May_2025.xlsx') and located in a year-named subfolder within your project (e.g., '2025/').")
        return # Exit if critical files are missing

    # Get basic summaries for the LLM (preprocessing happens inside get_basic_df_summary)
    summary_this_month_str = excel_utils.get_basic_df_summary(df_this_month)
    summary_last_month_str = excel_utils.get_basic_df_summary(df_last_month)

    prompt_filepath = ""
    info_filepath = "prompts/info.txt"
    info_content = ""

    if config.LANGUAGE == "Korean" or config.LANGUAGE == "한국어":
        greeting = "<p>안녕하세요 홍선님, <br><br>이번 달 가계부 정리 내용을 토대로 분석한 결과입니다.<br></p>" 
        prompt_filepath = "prompts/prompt_kr.txt"
    elif config.LANGUAGE == "English":
        greeting = "<p>Hi Sean, <br><br>Here is your budget analysis for this month.<br></p>"
        prompt_filepath = "prompts/prompt_en.txt"

    
    # Try to read the content of the selected prompt and info files
    try:
        with open(prompt_filepath, 'r', encoding='utf-8') as file:
            raw_prompt_template = file.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompt_filepath}. Please ensure it exists.")
        return # Exit if the prompt file is missing
    
    
    try:
        with open(info_filepath, 'r', encoding='utf-8') as file:
            info_content = file.read()
    except FileNotFoundError:
        print(f"Warning: Info file not found at {info_filepath}. Proceeding without additional context.")
        # It's usually fine to continue if the info file is just supplemental


    full_prompt = raw_prompt_template.format(
        df_this_month=df_this_month,
        this_month_display=this_month_display,
        current_month_summary=summary_this_month_str,
        df_last_month=df_last_month,
        last_month_display=last_month_display,
        last_month_summary=summary_last_month_str,
    )

    combined_raw_prompt = f"{info_content}\n\n---\n\n{full_prompt}"

    analysis_text = gemini_utils.get_gemini_response(combined_raw_prompt, model_name=config.GEMINI_MODEL)
    
    if analysis_text == "Error: Could not get analysis from Gemini.":
        print("Failed to get analysis from Gemini. Please check API key, network, or usage limits.")
        return # Exit if analysis failed

    # Output Handling (Email vs. Print) ---
    if config.EMAIL_SEND.lower() == "true":
        print("\n--- Sending Email Report ---")
        email_subject = f'{this_month_display} Budget Analysis Report'

        # Convert Markdown analysis_text to HTML
        analysis_html_body = markdown.markdown(analysis_text, extensions=['markdown.extensions.tables']) 

        # Concatenate the HTML greeting with the HTML analysis body
        email_body_final = greeting + analysis_html_body

        email_sent_success = email_utils.send_email(
            receiver_email=config.RECEIVER_EMAIL,
            subject=email_subject,
            body=email_body_final
        )
        if email_sent_success:
            print("Email report sent successfully!")
        else:
            print("Failed to send email report.")
    else:
        print(greeting + analysis_text)

    print("\n--- Budget Planner Analysis Script Finished ---")

if __name__ == "__main__":
    run_money_manager_analysis()