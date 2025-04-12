# DETECTING GIVEN E-MAIL IS SPAM OR NOT

# simple program of detecting email or not
"""def Email_spam(email_text):
    email_text = email_text.lower()
    spam_words = ["win" , "free" , " lucky member " , "price"]
    count_spam_words = sum(words in email_text for words in spam_words)
    if count_spam_words >= 2:
        return "spam"
    return "notspam" """
    
email_text ="You are a lucky member! Win a free price today!"
Email_spam(email_text)

# but its not a all time working program because in real life some real email are that in form like as win , price , lucky menebr
# according to this condition
def Check_email_spam(email_text1):
    email_text1 = email_text1.lower()
    spam_email_words = ["win" , "free" , " lucky member " , "price"]

    # Count how many spam keywords are present in the email
    counting_spam_words = sum(words in email_text1 for words in spam_email_words)
    
    # Check if the email contains any hyperlinks (common in spam)
    any_hyper_link = "http://" in email_text1

    # Check if the email contains any upper case letters (common in spam)
    any_upper_case_words = sum(1 for word in email_text1.split() if word.isupper()) > 3

    score_spam = 0

    if counting_spam_words >= 2:
        score_spam += 1

    if any_hyper_link >= 1:
        score_spam += 1

    if any_upper_case_words >= 2:
        score_spam += 1

    if score_spam >= 2:
        return "spam"
    return "notspam"


email_text1 = "You are a LUCKY member! win a free price today! click on link http:// you can WIN a PRICE "
Check_email_spam(email_text1)
