import win32com.client as win32
import pathlib, time
d = pathlib.Path(__file__).parent
docx = str(d / "DentScan_Report.docx")
pdf  = str(d / "DentScan_Report.pdf")
word = win32.Dispatch("Word.Application")
word.Visible = False
doc = word.Documents.Open(docx)
# update all fields (TOC, PAGE, etc.) — TOC ก่อน
for _ in range(2):
    doc.Fields.Update()
    if doc.TablesOfContents.Count:
        doc.TablesOfContents(1).Update()
    time.sleep(0.5)
doc.Save()
doc.ExportAsFixedFormat(pdf, 17)  # 17 = wdExportFormatPDF
doc.Close(False)
word.Quit()
print("PDF saved:", pdf)
