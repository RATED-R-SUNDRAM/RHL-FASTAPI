import re,os
import json
from pathlib import Path
from docx import Document
from PyPDF2 import PdfReader

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)


# ------------------------------
# 2. Extract text from DOCX
# ------------------------------
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_abbreviations(text):
    """
    Extracts abbreviations like:
    - BP (Blood Pressure)
    - b.p. (Blood Pressure)
    - COPD (Chronic Obstructive Pulmonary Disease)
    """
    abbr_dict = {}

    # Match: ABC (Full form)
    pattern = re.compile(r"\b([A-Za-z\.]{2,10})\s*\(([^)]+)\)")
    
    for match in pattern.finditer(text):
        abbr = match.group(1).replace(".", "").lower().strip()
        full_form = match.group(2).strip()

        if 2 <= len(abbr) <= 10 and full_form:
            abbr_dict[abbr] = full_form

    return abbr_dict


# ------------------------------
# 4. Build dictionary from multiple files
# ------------------------------
def build_abbreviation_dict(files, save_path="abbr_dict.json"):
    final_dict = {}

    for file in files:
        print("processing :",file)
        ext = Path(file).suffix.lower()

        if ext == ".pdf":
            text = extract_text_from_pdf(file)
        elif ext == ".docx":
            text = extract_text_from_docx(file)
        else:
            print(f"Skipping unsupported file: {file}")
            continue

        abbrs = extract_abbreviations(text)
        final_dict.update(abbrs)

    # Save clean JSON
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_dict, f, indent=2, ensure_ascii=False)

    print(f"✅ Abbreviation dictionary saved to {save_path}")
    return final_dict


# ------------------------------
# 5. Expand abbreviations in query using dictionary
# ------------------------------
def expand_abbreviations_in_text(query, abbr_dict):
    """
    Replace abbreviations in query with full form if found in dict.
    """
    words = query.split()
    expanded = []

    for word in words:
        key = word.lower().replace(".", "")
        if key in abbr_dict:
            expanded.append(abbr_dict[key])
        else:
            expanded.append(word)

    return " ".join(expanded)

path = "D:/Documents/RHL-RAG-PROJECT/FILES"
arr=[f"{path}/{i}" for i in os.listdir(r"D:\Documents\RHL-RAG-PROJECT\FILES")]


abbr_dict = build_abbreviation_dict(arr)

query = "Patient has COPD and high BP."
expanded_query = expand_abbreviations_in_text(query, abbr_dict)

print("\nOriginal:", query)
print("Expanded:", expanded_query)