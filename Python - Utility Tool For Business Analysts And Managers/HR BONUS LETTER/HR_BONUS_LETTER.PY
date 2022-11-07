from pathlib import Path
import pandas as pd  
from docxtpl import DocxTemplate 

base_dir = Path(__file__).parent

word_template_path = base_dir / "TEMPLATE_BONUS_LETTER.docx"

excel_path = base_dir / "Employees.xlsx"

output_dir = base_dir / "OUTPUT"


# Convert Excel sheet to pandas dataframe
df = pd.read_excel(excel_path, sheet_name="Sheet1")

# Iterate over each row in df and render word document
for record in df.to_dict(orient="records"):

    # print(f"The record value is :{record}")
    doc = DocxTemplate(word_template_path)
    doc.render(record)
    output_path = output_dir / f"{record['NAME']}-BONUS-LETTER.docx"
    doc.save(output_path)
