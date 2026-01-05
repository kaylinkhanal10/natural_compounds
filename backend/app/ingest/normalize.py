
import re

def normalize_key(s):
    if not isinstance(s, str):
        return str(s) if s is not None else ""
    return s.strip().lower().replace(" ", "_")

def simple_title_case(s):
    if not isinstance(s, str):
        return ""
    # Normalize multiple spaces
    s = re.sub(r'\s+', ' ', s.strip())
    # Title case
    return s.title()

def clean_float(val):
    try:
        return float(val)
    except:
        return None

def clean_int(val):
    try:
        return int(val)
    except:
        return None
