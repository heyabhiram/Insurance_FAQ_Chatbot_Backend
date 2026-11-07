import re

def normalize_text(text: str) -> str:
    """
    Normalizes input text by converting to lowercase, removing redundant whitespace,
    and expanding common contractions.
    """
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower().strip()

    # Expand standard contractions
    contractions = {
        r"\bcan't\b": "cannot",
        r"\bwon't\b": "will not",
        r"\bdon't\b": "do not",
        r"\bdid't\b": "did not",
        r"\bdidn't\b": "did not",
        r"\bisn't\b": "is not",
        r"\baren't\b": "are not",
        r"\bwhat's\b": "what is",
        r"\bthat's\b": "that is",
        r"\bhow's\b": "how is",
        r"\bit's\b": "it is",
        r"\bi've\b": "i have",
        r"\bi'm\b": "i am",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)

    # Clean multiple spaces and non-standard whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_query_for_search(text: str) -> str:
    """
    Cleans user query for embedding search by removing non-alphanumeric punctuation
    except meaningful characters like hyphens.
    """
    normalized = normalize_text(text)
    # Remove special punctuation like ? ! , . ; : " '
    cleaned = re.sub(r"[?!,;:\(\)\[\]\{\}\"\']", " ", normalized)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def is_greeting(text: str) -> bool:
    """
    Checks if the user input is purely a conversational greeting
    (e.g., 'Hello', 'Hai', 'Hiii', 'hi', 'heyy', 'hey there', 'good morning', etc.)
    without containing a substantive domain query.
    """
    if not text:
        return False
    norm = normalize_text(text)
    # Remove common punctuation
    cleaned = re.sub(r"[!?.~,;:]+", "", norm).strip()
    # Collapse excessive repeating characters (e.g., 'hiiii' -> 'hii', 'heyyyy' -> 'heyy')
    collapsed = re.sub(r"([a-z])\1{2,}", r"\1\1", cleaned)

    greeting_pattern = (
        r"^(?:h+[eia]+y*|h+a+i+|h+e+l+o+|hiya|hola|howdy|namaste|sup|yo|"
        r"good\s+(?:morning|afternoon|evening|day)|greetings)"
        r"(?:\s+(?:there|bot|assistant|drivesure|all|team|friend))?$"
    )
    return bool(re.match(greeting_pattern, collapsed))


