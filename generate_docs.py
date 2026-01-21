import os
import sys

# List of artifacts to include
# Note: Paths are hardcoded based on current knowledge; in a real app, we'd pass them in or find them.
ARTIFACT_DIR = "/Users/raju/.gemini/antigravity/brain/8980c6ba-e651-415b-817f-79af66cdf265"
FILES = [
    "presentation_notes.md",
    "network_slm_design.md",
    "implementation_plan.md"
]
OUTPUT_FILENAME = "Project_Documentation"

def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not read {path}: {e}")
        return f"*(File {os.path.basename(path)} missing)*"

def generate_html(content):
    """
    Wraps content in a styled HTML template for cleaner printing.
    """
    try:
        import markdown
    except ImportError:
        print("Error: 'markdown' library not installed.")
        return None

    html_body = markdown.markdown(content, extensions=['fenced_code', 'tables', 'toc'])
    
    css = """
    <style>
        body { font-family: 'Helvetica', 'Arial', sans-serif; line-height: 1.6; max-width: 800px; margin: auto; padding: 20px; }
        h1 { color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; page-break-before: always; }
        h1:first-of-type { page-break-before: avoidance; }
        pre { background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }
        code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-family: 'Courier New', monospace; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .page-break { page-break-after: always; }
    </style>
    """
    
    return f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Project Documentation</title>
        {css}
    </head>
    <body>
        {html_body}
    </body>
    </html>
    """

def convert_to_pdf(html_content, output_path):
    try:
        from xhtml2pdf import pisa
    except ImportError:
        print("xhtml2pdf not installed. Skipping PDF generation.")
        return False
        
    try:
        with open(output_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)
        return not pisa_status.err
    except Exception as e:
        print(f"PDF Generation failed: {e}")
        return False

def main():
    combined_md = "# Project Documentation\n\nGenerated from Workspace Artifacts.\n\n"
    
    for fname in FILES:
        fpath = os.path.join(ARTIFACT_DIR, fname)
        title = fname.replace(".md", "").replace("_", " ").title()
        content = read_file(fpath)
        
        # Add visual separator and title
        combined_md += f"\n\n# {title}\n\n"
        combined_md += content
        combined_md += "\n\n<div class='page-break'></div>\n"

    # 1. Generate HTML
    html_content = generate_html(combined_md)
    if not html_content:
        return

    # 2. Save HTML (Back up)
    html_path = f"{OUTPUT_FILENAME}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated HTML: {html_path}")

    # 3. Try PDF
    pdf_path = f"{OUTPUT_FILENAME}.pdf"
    print("Attempting PDF conversion...")
    success = convert_to_pdf(html_content, pdf_path)
    
    if success:
        print(f"Successfully created PDF: {pdf_path}")
    else:
        print(f"PDF creation failed. Please find the HTML file at {html_path} and print it to PDF.")

if __name__ == "__main__":
    main()
