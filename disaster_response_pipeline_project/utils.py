import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)


def tokenize(text):
    """
    Tokenize and lemmatize text, replacing URLs with a placeholder.

    Args:
        text (str): Raw text to process.

    Returns:
        list: Cleaned, lemmatized tokens.
    """
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    for url in re.findall(url_regex, text):
        text = text.replace(url, "urlplaceholder")

    lemmatizer = WordNetLemmatizer()
    return [
        lemmatizer.lemmatize(token).lower().strip()
        for token in word_tokenize(text)
    ]
