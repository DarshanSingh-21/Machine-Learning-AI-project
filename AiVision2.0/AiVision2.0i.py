import os
import io
import sys
import time
import re
from dotenv import load_dotenv

# SDK Engine Imports
from google import genai
from google.genai import types
import google.genai.errors
from groq import Groq 

import pandas as pd
import pdfplumber
from pydantic import BaseModel, Field
from pypdf import PdfReader 
from pdf2image import convert_from_path
from openpyxl import Workbook
from docx import Document as DocxWriter

# =========================================================================
# 1. DUAL ENVIRONMENT CONFIGURATION & BOOTSTRAP (WITH COHESIVE FALLBACKS)
# =========================================================================

load_dotenv()

gemini_key = os.environ.get("GEMINI_API_KEY")
groq_key = os.environ.get("GROQ_API_KEY")

# Primary and Secondary text models assigned via environment variables with hardcoded fallbacks
PRIMARY_TEXT = os.environ.get("PRIMARY_TEXT_MODEL", "llama-3.3-70b-versatile")
BACKUP_TEXT = os.environ.get("BACKUP_TEXT_MODEL", "gemma2-9b-it")

# Vision model orchestrations - Configured to use Gemini 3.1 Flash Lite as the automated redundant failover target
PRIMARY_VISION = os.environ.get("PRIMARY_VISION_MODEL", "gemini-2.5-flash")
BACKUP_VISION = os.environ.get("BACKUP_VISION_MODEL", "gemini-3.1-flash-lite")

if not gemini_key:
    print("❌ ERROR: Missing GEMINI_API_KEY in your .env file for Phase 3 Vision.")
    sys.exit(1)

if not groq_key:
    print("❌ ERROR: Missing GROQ_API_KEY in your .env file to run the Text layers.")
    sys.exit(1)

google_client = genai.Client(api_key=gemini_key)
groq_client = Groq(api_key=groq_key)

print("🔌 Core Engines Booted successfully. [Groq (Text LLM) + Google GenAI (Vision Matrix)]")

# =========================================================================
# 2. PURE METRICS-ONLY DATA SCHEMA FOR IMAGE ELEMENT
# =========================================================================

class SpatialDimensionRow(BaseModel):
    feature_name: str = Field(description="Name of entry")
    length: str = Field(description="Length measurement with units")
    breadth: str = Field(description="Breadth measurement with units")
    height: str = Field(description="Height measurement with units")
    calculated_area: str = Field(description="Calculated area metric layout footprint")

class PureDimensionalResult(BaseModel):
    items: list[SpatialDimensionRow]

# =========================================================================
# 3. SMART RETRY / MULTI-MODEL FALLBACK HANDLING MECHANISM
# =========================================================================

def execute_text_with_fallback(prompt: str) -> str:
    """Tries the primary text model on Groq first. If it encounters a rate limit

    or quota error, it falls back to the secondary text model.
    """
    try:
        print(f"🧠 Attempting primary Text analysis using: {PRIMARY_TEXT}")
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an elite research systems analyst."},
                {"role": "user", "content": prompt}
            ],
            model=PRIMARY_TEXT,
            temperature=0.3,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower() or "limit" in str(e).lower():
            print(f"⚠️ [RATE LIMIT / ERROR] Primary model {PRIMARY_TEXT} failed. Swapping to backup...")
            print(f"🔄 Routing request to backup Text model: {BACKUP_TEXT}")
            try:
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an elite research systems analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    model=BACKUP_TEXT,
                    temperature=0.3,
                )
                return chat_completion.choices[0].message.content.strip()
            except Exception as backup_err:
                print(f"❌ Critical: Backup text model also failed: {backup_err}")
                return None
        else:
            raise e

def execute_vision_with_fallback(model_name: str, contents_payload, config_payload):
    """Executes content generation via Google SDK. If a 429 resource exhaustion

    or request threshold failure triggers, it automatically drops down to the
    Gemini 3.1 Flash Lite backup layer.
    """
    try:
        response = google_client.models.generate_content(
            model=model_name,
            contents=contents_payload,
            config=config_payload
        )
        return response
    except (google.genai.errors.APIError, Exception) as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
            if model_name == PRIMARY_VISION:
                print(f"\n⚠️ [RATE LIMIT EXHAUSTED] {PRIMARY_VISION} tier limit hit. Failing over to backup processing standard...")
                print(f"🔄 Contacting fallback Vision model: {BACKUP_VISION} (Gemini 3.1 Flash Lite Engine)...")
                return execute_vision_with_fallback(BACKUP_VISION, contents_payload, config_payload)
        raise e

# =========================================================================
# 4. SAFETY ASSURANCE WORKERS
# =========================================================================

def check_pdf_has_images(pdf_path: str, target_page_idx: int = 0) -> bool:
    try:
        reader = PdfReader(pdf_path)
        if target_page_idx >= len(reader.pages):
            return False
        
        page = reader.pages[target_page_idx]
        if "/Resources" in page and "/XObject" in page["/Resources"]:
            xobjects = page["/Resources"]["/XObject"].get_object()
            for obj in xobjects:
                if xobjects[obj].get_object()["/Subtype"] == "/Image":
                    return True
        return False
    except Exception:
        return False

def extract_raw_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    extracted_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text.append(text)
    return "\n".join(extracted_text).strip()

def save_analysis_to_docx(analysis_content: str, docx_output_path: str, report_title: str):
    doc = DocxWriter()
    doc.add_heading(report_title, level=0)
    paragraphs = analysis_content.split('\n')
    for p in paragraphs:
        clean_p = p.strip()
        if clean_p.startswith("###"):
            doc.add_heading(clean_p.replace("###", "").strip(), level=3)
        elif clean_p.startswith("##"):
            doc.add_heading(clean_p.replace("##", "").strip(), level=2)
        elif clean_p.startswith("#"):
            doc.add_heading(clean_p.replace("#", "").strip(), level=1)
        elif clean_p:
            doc.add_paragraph(clean_p)
    doc.save(docx_output_path)
    print(f"✨ Text report written to Word document: {docx_output_path}")

def run_text_analysis_layer(raw_text: str) -> str:
    prompt = (
        "Perform a rigorous thematic analysis of the provided text. Do not summarize briefly.\n"
        "1. Generate a comprehensive executive summary outlining overarching objectives.\n"
        "2. Isolate the top 5 most critical technical trends, system metrics, or structural data insights.\n"
        "3. Elaborate deeply on each insight—provide domain context and actionable next steps.\n\n"
        f"Raw Extracted Text Context:\n{raw_text}"
    )
    return execute_text_with_fallback(prompt)

def analyze_canvas_dimensions_native(img_bytes, img_w, img_h):
    prompt_base = (
        f"The layout image boundaries are exactly {img_w} wide by {img_h} high in pixels.\n"
        "Extract ONLY numerical dimensions from the attached page layout view. DO NOT provide any text descriptions, notes, or explanations.\n\n"
        "STRICT DATA REGISTRATION RULES:\n"
        "1. FIRST ENTRY: Always register the total overall image canvas layout footprint under feature_name='Overall Image Canvas'.\n"
        "2. INTERNAL SYMMETRIC OBJECTS: If there are distinct symmetrical geometric shapes located inside the image boundary, map their specific structural boundaries as additional rows.\n"
        "3. NO TEXT/DESCRIPTIONS: Populate ONLY raw measurements (Length, Breadth, Height, Area with units) for each tracked item."
    )

    # Mode A Implementation with Fallback Multi-Model Wrapping
    def schema_call():
        print(f"🛰️ Streaming layout matrix to Vision Engine [Mode A: Schema Extraction]...")
        response = execute_vision_with_fallback(
            model_name=PRIMARY_VISION,
            contents_payload=[
                types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'),
                prompt_base + "\nStrictly adhere to the requested structural JSON response schema."
            ],
            config_payload=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PureDimensionalResult,
                temperature=0.0
            )
        )
        return response.parsed.items if (response.parsed and hasattr(response.parsed, 'items')) else None

    try:
        result = schema_call()
        if result:
            return result
    except Exception as e:
        print(f"⚠️ Mode A hit an operational fault: {e}. Moving to Mode B fallback layout directly...")

    # Mode B Text Parsing Fallback
    def fallback_call():
        print(f"🛰️ Streaming layout matrix to Vision Engine [Mode B: Text Parsing Fallback]...")
        fallback_prompt = (
            prompt_base + "\n"
            "Output your findings as a clean table matching this style, separation with a vertical bar '|'.\n"
            "Example format:\n"
            "Overall Image Canvas | 1200px | 800px | 0px | 960000 sq px\n"
            "Internal Box | 400px | 400px | 0px | 160000 sq px"
        )
        
        response = execute_vision_with_fallback(
            model_name=PRIMARY_VISION,
            contents_payload=[
                types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'),
                fallback_prompt
            ],
            config_payload=types.GenerateContentConfig(temperature=0.0)
        )
        
        text_lines = response.text.strip().split('\n')
        parsed_items = []
        for line in text_lines:
            if "|" in line and "Feature Name" not in line and "---" not in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4:
                    while len(parts) < 5:
                        parts.append("0")
                    
                    class RowObject:
                        def __init__(self, f, l, b, h, a):
                            self.feature_name = f
                            self.length = l
                            self.breadth = b
                            self.height = h
                            self.calculated_area = a
                    parsed_items.append(RowObject(parts[0], parts[1], parts[2], parts[3], parts[4]))
        return parsed_items if len(parsed_items) > 0 else None

    return fallback_call()

# =========================================================================
# 5. MASTER INTEGRATED PIPELINE ORCHESTRATOR
# =========================================================================

def run_comprehensive_individual_pipeline(pdf_path: str):
    local_script_folder = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    file_base = os.path.splitext(os.path.basename(pdf_path))[0]

    text_output_file = os.path.join(local_script_folder, f"{file_base}_Text_Deep_Analysis.docx")
    tables_output_file = os.path.join(local_script_folder, f"{file_base}_Extracted_Tables.xlsx")
    vision_output_file = os.path.join(local_script_folder, f"{file_base}_Image_Dimensions.xlsx")

    # ---------------------------------------------------------------------
    # PHASE 1: Text Intelligence Generation
    # ---------------------------------------------------------------------
    if os.path.exists(text_output_file):
        print(f"ℹ️ Skipping PHASE 1: '{os.path.basename(text_output_file)}' already exists.")
    else:
        print("⚡ [PHASE 1] Scanning structural text layers...")
        raw_document_text = extract_raw_pdf_text(pdf_path)
        if raw_document_text:
            text_analysis_content = run_text_analysis_layer(raw_document_text)
            if text_analysis_content:
                save_analysis_to_docx(text_analysis_content, text_output_file, "CAiVision Advanced Document Text Analysis")
        else:
            print("ℹ️ No text found; skipping Word artifact creation.")

    # ---------------------------------------------------------------------
    # PHASE 2: Data Table Isolation (SAFEGUARDED AGAINST EMPTY GENERATION)
    # ---------------------------------------------------------------------
    if os.path.exists(tables_output_file):
        print(f"ℹ️ Skipping PHASE 2: '{os.path.basename(tables_output_file)}' already exists.")
    else:
        print("\n⚡ [PHASE 2] Auditing layout matrix coordinates for data grids...")
        
        table_extraction_settings = {
            "vertical_strategy": "lines", "horizontal_strategy": "lines",  
            "snap_tolerance": 4, "join_tolerance": 4, "edge_min_length": 15, "intersection_tolerance": 4
        }
        
        table_counter = 0
        valid_tables_found = {}

        with pdfplumber.open(pdf_path) as pdf:
            for idx, page in enumerate(pdf.pages):
                extracted_tables = page.extract_tables(table_settings=table_extraction_settings)
                
                for t_idx, current_table in enumerate(extracted_tables):
                    if not current_table or len(current_table) <= 1 or len(current_table[0]) <= 1:
                        continue

                    table_counter += 1
                    sheet_title = f"Page{idx+1}_Table{table_counter}"
                    processed_rows = []
                    
                    for data_row in current_table:
                        clean_row = []
                        for val in data_row:
                            if val is None:
                                clean_row.append("")
                            else:
                                clean_val = re.sub(r'\s+', ' ', str(val)).strip()
                                clean_row.append(clean_val)
                        processed_rows.append(clean_row)
                    
                    valid_tables_found[sheet_title] = processed_rows
                        
        if len(valid_tables_found) > 0:
            wb_tables = Workbook()
            default_sheet = wb_tables.active
            wb_tables.remove(default_sheet)
            for sheet_name, table_rows in valid_tables_found.items():
                ws_table = wb_tables.create_sheet(title=sheet_name)
                ws_table.views.sheetView[0].showGridLines = True
                for row in table_rows:
                    ws_table.append(row)
            wb_tables.save(tables_output_file)
            print(f"✨ SUCCESS! Isolated {len(valid_tables_found)} true structured tables: {tables_output_file}")
        else:
            print("ℹ️ Safeguard Tripped: No structural data tables found. Skipping table Excel creation entirely.")

    # ---------------------------------------------------------------------
    # PHASE 3: Safeguarded Canvas Vision Mapping Engine Layer
    # ---------------------------------------------------------------------
    if os.path.exists(vision_output_file):
        print(f"ℹ️ Skipping PHASE 3: '{os.path.basename(vision_output_file)}' already exists.")
    else:
        print("\n⚡ [PHASE 3] Checking file structural layers for image content...")
        
        if not check_pdf_has_images(pdf_path, target_page_idx=0):
            print("ℹ️ Safeguard Tripped: No graphic image objects found on page 1. Skipping Phase 3 entirely.")
        else:
            print("📸 Image files discovered! Launching vision spatial processing engine...")
            try:
                poppler_bin_path = r"D:\Poppler\Release-26.02.0-0\poppler-26.02.0\Library\bin"

                if os.path.exists(poppler_bin_path):
                    pages = convert_from_path(pdf_path, first_page=1, last_page=1, poppler_path=poppler_bin_path)
                else:
                    print(f"❌ Error: Invalid Poppler directory configuration mapping: '{poppler_bin_path}'")
                    return

                if pages:
                    page_image = pages[0].convert("RGB")
                    img_w, img_h = page_image.size
                    print(f"📐 Target Image Canvas Boundary Bounds: {img_w}x{img_h}px")
                    
                    working_img = page_image.copy()
                    working_img.thumbnail((1024, 1024))
                    
                    img_byte_arr = io.BytesIO()
                    working_img.save(img_byte_arr, format='JPEG', quality=85)
                    compressed_bytes = img_byte_arr.getvalue()
                    
                    vision_rows = analyze_canvas_dimensions_native(compressed_bytes, img_w, img_h)
                    
                    if vision_rows:
                        wb_vision = Workbook()
                        ws_vision = wb_vision.active
                        ws_vision.title = "Dimensions Ledger"
                        ws_vision.views.sheetView[0].showGridLines = True
                        ws_vision.append(["Feature Name", "Length", "Breadth", "Height", "Estimated Area"])
                        for item in vision_rows:
                            ws_vision.append([item.feature_name, item.length, item.breadth, item.height, item.calculated_area])
                        wb_vision.save(vision_output_file)
                        print(f"✨ SUCCESS! Image metrics sheet compiled: {vision_output_file}")
                    else:
                        print("❌ Error: Vision Extraction layers failed to return values.")
            except Exception as e:
                print(f"⚠️ Vision engine execution error: {e}")

    print("\n🏁 [SUCCESS] Pipeline execution evaluated completely.")


if __name__ == "__main__":
    print("====================================================")
    print("      CAiVision Advanced Hybrid Document Pipeline  ")
    print("====================================================\n")
    
    while True:
        user_input = input("📥 Enter the name of your PDF file (e.g., sample_data.pdf): ").strip()
        if user_input and not user_input.lower().endswith('.pdf'):
            user_input += '.pdf'
            
        if os.path.exists(user_input):
            target_pdf = user_input
            print(f"✅ Found target file: '{target_pdf}'")
            break
        else:
            print(f"❌ Error: The file '{user_input}' could not be found in this directory.\n")

    run_comprehensive_individual_pipeline(target_pdf)