from fpdf import FPDF

text = """Cardiovascular Services Section
The global cardiovascular stress test (CPT-93015), which includes continuous 12-lead ECG monitoring, exercise or pharmacologic stress, physician supervision, and the final interpretation report, is covered up to a maximum limit of $300.00. Please note that diagnostic procedures billed under CPT-93015 often require prior authorization from the primary care provider."""

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, txt=text)
pdf.output("data/sample_policy.pdf")
