import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

def send_email(receiver_email, subject, body):
    """
    Sends an email using environment variables for sender credentials and SMTP settings.

    Args:
        receiver_email (str): The email address of the recipient.
        subject (str): The subject line of the email.
        body (str): The content of the email.
        is_html (bool): If True, the body will be sent as HTML; otherwise, as plain text.

    Returns:
        bool: True if the email was sent successfully, False otherwise.
    """
    load_dotenv() # Load environment variables
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = os.getenv("SMTP_PORT")

    if not all([sender_email, sender_password, smtp_server, smtp_port]):
        print("Error: Missing one or more email environment variables (SENDER_EMAIL, SENDER_PASSWORD, SMTP_SERVER, SMTP_PORT).")
        print("Please ensure your .env file is correctly configured.")
        return False

    try:
        smtp_port = int(smtp_port)
    except ValueError:
        print(f"Error: Invalid SMTP_PORT '{smtp_port}'. Must be an integer.")
        return False

    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP_SSL(smtp_server, smtp_port) # Use SMTP_SSL for port 465 (Gmail default)
        
        # Log in to the email account
        server.login(sender_email, sender_password)

        # Send the email
        server.send_message(msg)
        print(f"Email sent successfully to {receiver_email}!")
        return True

    except smtplib.SMTPAuthenticationError:
        print("Error: SMTP Authentication failed. Check your SENDER_EMAIL and SENDER_PASSWORD.")
        print("For Gmail, you might need to use an App Password if 2FA is enabled.")
        return False
    except smtplib.SMTPConnectError as e:
        print(f"Error: Could not connect to SMTP server '{smtp_server}:{smtp_port}'. Is the server address correct and port open? {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while sending email: {e}")
        return False
    finally:
        if 'server' in locals() and server:
            server.quit() # Always close the connection

# --- Test functionality when email_utils.py is run directly ---
if __name__ == "__main__":
    print("--- Testing email_utils.py functions ---")
    load_dotenv()
    sender_email = os.getenv("SENDER_EMAIL")

    test_receiver_email = sender_email
    test_subject = "Testing Sending Email - Python"
    test_body_plain = "Hello,\n\nThis is a test email from your script."

    print("\nAttempting to send plain text email...")
    success_plain = send_email(test_receiver_email, test_subject, test_body_plain)
    print(f"Plain text email sent status: {success_plain}")

    print("\n--- Finished email_utils.py tests ---")