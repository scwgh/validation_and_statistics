import smtplib
from email.message import EmailMessage

def send_bulk_emails(sender, app_password, subject, content, recipient_list):
    for email in recipient_list:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = email
        msg.set_content(content)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, app_password)
            smtp.send_message(msg)
