import os
import tempfile
from io import BytesIO
from PIL import Image as PILImage
from PIL import ImageEnhance
from PIL import ImageFilter
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
from markdown2 import markdown
import re
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import HexColor
from typing import Optional

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------
st.set_page_config(
    page_title="Medical Imaging AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# Theme Configuration
# ----------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

# ----------------------------------------------------------
# Custom CSS for UI
# ----------------------------------------------------------
def load_css():
    theme = st.session_state.theme
    
    # ----------------Custom Color Palettes ----------------
    if theme == "dark":
        # Dark Mode: Deep Blue/Navy Medical Theme
        bg_color = "#0B1120"            # Very dark navy
        card_bg = "#1E293B"             # Slate 800
        text_color = "#F1F5F9"          # Slate 100
        secondary_text = "#94A3B8"      # Slate 400
        accent_color = "#38BDF8"        # Sky 400
        border_color = "#334155"        # Slate 700
        metadata_bg = "#0F172A"         # Slate 900
        upload_bg = "#1E293B"           # Match card
        input_bg = "#0F172A"            # Darker for inputs
        shadow_color = "rgba(0, 0, 0, 0.4)"
        
    else:
        # Light Mode: Clean, Clinical, High Contrast (Matching Reference)
        bg_color = "#F5F5F5"            # Light grey background
        card_bg = "#FFFFFF"             # Pure White cards
        text_color = "#2C3E50"          # Dark blue-grey text (high contrast)
        secondary_text = "#6C757D"      # Medium grey for secondary text
        accent_color = "#0EA5E9"        # Bright sky blue (matching reference)
        border_color = "#E5E7EB"        # Light border
        metadata_bg = "#E8F4F8"         # Light blue tint for metadata
        upload_bg = "#FFFFFF"           # White
        input_bg = "#FFFFFF"            # White
        shadow_color = "rgba(0, 0, 0, 0.08)"

    # ---------------- CSS Generation ----------------
    css = f"""
    <style>
    /* Global Reset & Background */
    .stApp {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
    }}
    
    /* ---------------- TEXT STYLING ---------------- */
    /* Headers */
    h1, h2, h3, h4, h5, h6, .stHeading {{
        color: {text_color} !important;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }}
    
    /* Standard Text & Markdown */
    .stMarkdown p, .stMarkdown li, .stMarkdown span, .stMarkdown div {{
        color: {text_color} !important;
    }}
    
    /* Captions, Hints, Small Text */
    .stCaption, .stMarkdown small, .upload-container p, .stFileUploader label, small {{
        color: {secondary_text} !important;
    }}
    
    /* Strong/Bold Text */
    strong, b {{
        color: {text_color} !important;
        font-weight: 600;
    }}

    /* ---------------- CARDS & CONTAINERS ---------------- */
    .custom-card, .analysis-container {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px {shadow_color};
        color: {text_color} !important;
        margin-bottom: 1rem;
    }}
    
    /* Fix for markdown headers inside analysis container */
    .analysis-container h1,
    .analysis-container h2,
    .analysis-container h3,
    .analysis-container h4 {{
        color: {text_color} !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
    }}
    
    .analysis-container h3 {{
        font-size: 1.25rem !important;
        border-bottom: 2px solid {accent_color} !important;
        padding-bottom: 0.5rem !important;
    }}
    
    .metadata-card {{
        background-color: {metadata_bg} !important;
        border-left: 4px solid {accent_color} !important;
        padding: 10px;
        border-radius: 6px;
        margin-bottom: 8px;
        color: {text_color} !important;
        font-size: 0.95rem;
    }}

    /* ---------------- ENHANCED UPLOAD AREA ---------------- */
    /* Style the file uploader container */
    [data-testid="stFileUploader"] {{
        background-color: {upload_bg} !important;
        border: 2px dashed {border_color} !important;
        border-radius: 12px;
        padding: 2rem !important;
        transition: all 0.3s ease;
    }}
    
    /* Hover effect for drag-drop */
    [data-testid="stFileUploader"]:hover {{
        border-color: {accent_color} !important;
        background-color: {metadata_bg} !important;
    }}
    
    /* Style the upload section */
    [data-testid="stFileUploader"] section {{
        border: none !important;
        padding: 1rem;
    }}
    
    /* Center and style the upload button */
    [data-testid="stFileUploader"] button {{
        background-color: {accent_color} !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        margin: 1rem auto !important;
        display: block !important;
        transition: opacity 0.2s;
    }}
    
    [data-testid="stFileUploader"] button:hover {{
        opacity: 0.9 !important;
    }}
    
    /* Style the drag-drop text */
    [data-testid="stFileUploader"] section > div {{
        color: {text_color} !important;
        text-align: start !important;
    }}
    
    [data-testid="stFileUploader"] small {{
        color: {secondary_text} !important;
        display: block !important;
        text-align: center !important;
        margin-top: 0.5rem !important;
    }}
    
    [data-testid="stFileUploader"] label {{
        color: {text_color} !important;
    }}
    
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] {{
        color: {text_color} !important;
    }}
    
    [data-testid="stFileUploader"] p {{
        color: {text_color} !important;
    }}
    
    [data-testid="stImage"] img {{
        width: 100% !important;      
        height: 450px !important;    
        object-fit: contain !important; 
    }}
    
    [data-testid="stImage"] {{
        display: flex !important;
        justify-content: center !important;
    }}

    /* Image preview styling */
    .image-preview {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
    }}
    
    .image-preview img {{
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px {shadow_color};
    }}

    /* ---------------- CONTROLS & INPUTS ---------------- */
    /* Primary Buttons */
    .stButton button {{
        background-color: {accent_color} !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600;
        border-radius: 8px;
        transition: opacity 0.2s;
        box-shadow: 0 2px 4px {shadow_color};
        padding: 0.75rem 2rem !important;
    }}
    .stButton button:hover {{
        opacity: 0.9;
    }}
    
    /* Download Button (Primary Type) */
    [data-testid="stDownloadButton"] button {{
        background-color: {accent_color} !important;
        color: #FFFFFF !important;
    }}

    /* Text Inputs */
    .stTextInput input {{
        background-color: {input_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px;
    }}
    .stTextInput input:focus {{
        border-color: {accent_color} !important;
        box-shadow: 0 0 0 1px {accent_color} !important;
    }}
    
    /* ---------------- SIDEBAR ---------------- */
    [data-testid="stSidebar"] {{
        background-color: {card_bg} !important;
        border-right: 1px solid {border_color} !important;
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: {text_color} !important;
    }}
    
    /* ---------------- FOOTER ---------------- */
    .fixed-footer {{
        background-color: {card_bg} !important;
        color: {secondary_text} !important;
        border-top: 1px solid {border_color};
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        padding: 12px;
        text-align: center;
        z-index: 1000;
        font-size: 0.85rem;
        box-shadow: 0 -2px 10px {shadow_color};
    }}

    /* ---------------- MISC ---------------- */
    /* Alerts/Info Boxes text fix */
    .stAlert {{
        color: {text_color} !important;
    }}
    
    /* Info boxes */
    .stInfo {{
        background-color: {card_bg} !important;
        border-left: 4px solid {accent_color} !important;
        color: {text_color} !important;
    }}
    
    .stSuccess {{
        color: {text_color} !important;
    }}
    
    /* Scrollbar styling for Webkit */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: {bg_color}; 
    }}
    ::-webkit-scrollbar-thumb {{
        background: {border_color}; 
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {secondary_text}; 
    }}
    
        /* ---------------- LINKS ---------------- */
/* Force link colors everywhere */
a {{
    color: #0EA5E9 !important;
    text-decoration: none !important;
}}

a:hover {{
    color: #38BDF8 !important;
    text-decoration: underline !important;
}}

/* Markdown links */
.stMarkdown a, .stMarkdown a:link, .stMarkdown a:visited {{
    color: #0EA5E9 !important;
}}

.stMarkdown a:hover {{
    color: #38BDF8 !important;
}}

/* Analysis box links */
.analysis-results-box a {{
    color: #0EA5E9 !important;
}}

.analysis-results-box a:hover {{
    color: #38BDF8 !important;
}}

/* Custom card links */
.custom-card a, .analysis-container a {{
    color: #0EA5E9 !important;
}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
                  
load_css()

# ----------------------------------------------------------
# Session State Initialization
# ----------------------------------------------------------
if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state.GOOGLE_API_KEY = None
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "pdf" not in st.session_state:
    st.session_state.pdf = None

# ----------------------------------------------------------
# Sidebar where you can set API Key for Gemini AI
# ----------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🔑 API Configuration")
    if not st.session_state.GOOGLE_API_KEY:
        api_key = st.text_input("Google API Key", type="password", placeholder="Enter your API key...")
        st.caption("🔗 [Get your API key](https://aistudio.google.com/app/apikey)")
        if api_key:
            st.session_state.GOOGLE_API_KEY = api_key
            st.success("✅ API Key Saved!")
            st.rerun()
    else:
        st.success("✅ API Key Configured")
        if st.button("🔄 Reset API Key"):
            st.session_state.GOOGLE_API_KEY = None
            st.rerun()

# ----------------------------------------------------------
# Agent Tool Functions (callable by the agent)
# ----------------------------------------------------------
# Store context globally for tool access
_tool_context = {
    "dicom_bytes": None,
    "image": None,
    "dicom_metadata": None
}

# ----------------------------------------------------------
# Docom Helper Functions
# ----------------------------------------------------------

def get_dicom_metadata() -> str:
    """
    Extract and return DICOM metadata from the uploaded medical image.
    Use this tool when you need patient information from a DICOM file.
    
    Returns:
        Formatted string containing patient demographics and study information
    """
    if _tool_context["dicom_bytes"] is None:
        return "No DICOM file available. This image is not in DICOM format."
    
    metadata = extract_dicom_metadata_helper(_tool_context["dicom_bytes"])
    
    result = "DICOM Metadata:\n\n"
    for key, value in metadata.items():
        result += f"• {key}: {value}\n"
    
    return result

def extract_dicom_metadata_helper(file_bytes):
    """
    Extracts metadata with proper date formatting and specific medical fields.
    """
    try:
        ds = pydicom.dcmread(BytesIO(file_bytes), force=True)
        
        # 1. Generate Raw Tags (for the expander)
        all_tags = {}
        for elem in ds:
            if elem.tag.group == 0x7fe0: continue # Skip pixels
            key = elem.keyword if elem.keyword else str(elem.tag)
            val = str(elem.value)
            if len(val) > 100: val = val[:100] + "..."
            all_tags[key] = val

        # 2. Extract Specific Fields for the Report/Card
        # Helper to safely get attributes
        def get_val(tag, default="N/A"):
            val = getattr(ds, tag, default)
            return str(val).strip() if val else "N/A"

        # Format specific values
        dob = format_dicom_date(get_val("PatientBirthDate"))
        study_date = format_dicom_date(get_val("StudyDate"))
        
        weight = get_val("PatientWeight")
        if weight != "N/A":
            weight = f"{weight} kg"

        summary = {
            "Patient Name": get_val("PatientName").replace("^", " "),
            "Patient ID": get_val("PatientID"),
            "Age": get_val("PatientAge"),
            "DOB": dob,
            "Sex": get_val("PatientSex"),
            "Weight": weight,
            "Body Part": get_val("BodyPartExamined"),
            "Modality": get_val("Modality"),
            "Study Date": study_date,
            "Institution": get_val("InstitutionName"),
            "Physician": get_val("ReferringPhysicianName").replace("^", " "),
            "Description": get_val("StudyDescription")
        }

        # Filter out empty values (keep the list clean)
        summary = {k: v for k, v in summary.items() if v != "N/A" and v != ""}

        return summary, all_tags

    except Exception as e:
        return {"Error": str(e)}, {}
  
def dicom_to_pil_helper(file_bytes):
    try:
        ds = pydicom.dcmread(BytesIO(file_bytes), force=True)
        
        # 1. Get the pixel array
        pixel_array = ds.pixel_array.astype(float)
        
        # 2. Handle Photometric Interpretation (Invert if needed)
        # MONOCHROME1 means 0=White, so we invert it to standard 0=Black
        if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
            pixel_array = np.max(pixel_array) - pixel_array

        # 3. Apply standard DICOM Windowing (VOI LUT) if available
        # This uses the settings embedded by the radiologist/machine
        from pydicom.pixel_data_handlers.util import apply_voi_lut
        try:
            pixel_array = apply_voi_lut(ds.pixel_array, ds)
            pixel_array = pixel_array.astype(float)
        except Exception:
            pass # If VOI LUT fails, we fall back to auto-windowing below

        # 4. Handle 3D volumes (take the middle slice)
        if len(pixel_array.shape) == 3:
            mid_slice = pixel_array.shape[0] // 2
            pixel_array = pixel_array[mid_slice]

        # 5. AUTO-WINDOWING / CONTRAST ENHANCEMENT
        # Instead of using min/max, we clip the top/bottom 1% of pixels.
        # This removes outliers (like pure black air or bright metal) to fix contrast.
        if pixel_array.size > 0:
            p_low, p_high = np.percentile(pixel_array, (1, 99)) # 1st and 99th percentile
            if p_high > p_low:
                pixel_array = np.clip(pixel_array, p_low, p_high)
                pixel_array = (pixel_array - p_low) / (p_high - p_low)
            else:
                pixel_array = np.zeros_like(pixel_array) # Avoid division by zero
        
        # 6. Convert to standard 8-bit image
        pixel_array = (pixel_array * 255).astype(np.uint8)
        
        return PILImage.fromarray(pixel_array).convert("L")
    
    except Exception as e:
        print(f"Error converting DICOM: {e}")
        # Return a placeholder on error
        return PILImage.new("L", (512, 512), color=128)

def format_dicom_date(date_str):
    """Helper to convert DICOM YYYYMMDD to DD/MM/YYYY"""
    if not date_str or str(date_str) == "N/A":
        return "N/A"
    
    date_str = str(date_str).strip()
    
    # Standard DICOM date format is 8 chars (YYYYMMDD)
    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[6:8]}/{date_str[4:6]}/{date_str[0:4]}"
    
    return date_str

# ----------------------------------------------------------
# PDF Generation Helper Functions
# ----------------------------------------------------------
def generate_pdf_helper(image: PILImage, analysis_text: str, dicom_metadata: dict = None):
    """
    Generates a PDF with:
    1. Dynamic Title (Patient Name)
    2. Professional Metadata Table
    3. Small, Enhanced Image (No Title above it)
    4. Black Headings, Blue Links
    """
    buffer = BytesIO()

    # 1. Document Setup
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=45, leftMargin=45, topMargin=45, bottomMargin=45
    )

    # 2. Define Styles
    styles = getSampleStyleSheet()
    
    # Title Style
    title_style = ParagraphStyle(
        name='MedicalTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        alignment=1, # Center
        spaceAfter=20,
        textColor=colors.black,
        fontName="Helvetica-Bold"
    )

    # Section Header Style (Black)
    header_style = ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.black, # STRICTLY BLACK
        fontName="Helvetica-Bold",
        textTransform='uppercase'
    )

    # Normal Text
    normal_style = ParagraphStyle(
        name='MedicalNormal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.black
    )

    # Bullet Points
    bullet_style = ParagraphStyle(
        name='MedicalBullet',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        leftIndent=15,
        spaceAfter=2,
        textColor=colors.black
    )

    story = []

    # ---------------------------------------------------------
    # PART 1: DYNAMIC HEADER (Patient Name + Radiology Report)
    # ---------------------------------------------------------
    patient_name = "Medical Imaging"
    if dicom_metadata and "Patient Name" in dicom_metadata:
        patient_name = str(dicom_metadata["Patient Name"]).replace("^", " ").strip()
    
    story.append(Paragraph(f"{patient_name} Radiology Report", title_style))
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # PART 2: STRUCTURED PATIENT DATA
    # ---------------------------------------------------------
    if dicom_metadata:
        table_data = []
        for k, v in dicom_metadata.items():
            label = Paragraph(f"<b>{k}</b>", normal_style)
            value = Paragraph(str(v), normal_style)
            table_data.append([label, value])

        patient_table = Table(table_data, colWidths=[150, 380])
        
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(patient_table)
        story.append(Spacer(1, 20))

    # ---------------------------------------------------------
    # PART 3: SMALL ENHANCED IMAGE (No Title)
    # ---------------------------------------------------------
    try:
        # Note: Title removed as requested
        
        pdf_img = image.copy()
        if pdf_img.mode != 'L':
            pdf_img = pdf_img.convert('L')

        # Enhancement
        pdf_img = pdf_img.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(pdf_img)
        pdf_img = enhancer.enhance(1.2) 

        # Sizing Logic: SMALLER SIZE (Thumbnail style)
        # Max width set to 250px (approx 3 inches)
        max_width = 250 
        aspect = pdf_img.height / pdf_img.width
        new_height = int(max_width * aspect)
        
        # Hard Cap on Height (350px) to prevent vertical layout breaking
        if new_height > 350:
            new_height = 350
            max_width = int(new_height / aspect)

        img_buffer = BytesIO()
        pdf_img.resize((max_width, new_height), PILImage.LANCZOS).save(img_buffer, format="PNG", quality=100)
        img_buffer.seek(0)

        # Center the image
        story.append(RLImage(img_buffer, width=max_width, height=new_height))
        story.append(Spacer(1, 20))
    except Exception:
        story.append(Paragraph("<i>[Image Processing Error]</i>", normal_style))

    # ---------------------------------------------------------
    # PART 4: ANALYSIS CONTENT
    # ---------------------------------------------------------
    
    def format_text(text):
        # Bold: **text** -> <b>text</b>
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        
        # Links: [text](url) -> Blue, Underlined Link
        text = re.sub(
            r'\[(.*?)\]\((.*?)\)', 
            r'<a href="\2"><font color="blue"><u>\1</u></font></a>', 
            text
        )
        return text

    lines = analysis_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue

        formatted_line = format_text(line)

        # Headings (###) -> Black Section Header
        if line.startswith('###') or line.startswith('##'):
            clean_head = line.replace('#', '').strip()
            story.append(Paragraph(clean_head, header_style))
        
        # Bullets
        elif line.startswith('- ') or line.startswith('* '):
            clean_item = formatted_line[1:].strip()
            story.append(Paragraph(f"• {clean_item}", bullet_style))
        
        # Normal Text
        else:
            story.append(Paragraph(formatted_line, normal_style))
        
        story.append(Spacer(1, 4))

    # Footer
    story.append(Spacer(1, 30))
    disclaimer_style = ParagraphStyle(
        name='Disclaimer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=1
    )
    story.append(Paragraph("<b>DISCLAIMER:</b> AI-generated report for educational purposes only.", disclaimer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_pdf_report(analysis_text: str) -> str:
    """
    Generate a professional PDF radiology report from your analysis.
    Call this tool with your complete markdown-formatted analysis after you finish the medical image analysis.
    
    Args:
        analysis_text: Your complete medical analysis in markdown format
        
    Returns:
        Confirmation message that PDF has been generated
    """
    print("🔧 DEBUG: generate_pdf_report tool was called!")  # Debug line
    
    if _tool_context["image"] is None:
        return "Error: No image available for PDF generation."
    
    try:
        pdf_buffer = generate_pdf_helper(
            _tool_context["image"],
            analysis_text,
            _tool_context["dicom_metadata"]
        )
        
        # Store PDF in session state for download
        st.session_state.pdf = pdf_buffer
        
        print("🔧 DEBUG: PDF stored in session state successfully!")  # Debug line
        
        return "PDF report successfully generated! The user can now download it."
    
    except Exception as e:
        print(f"🔧 DEBUG: PDF generation error: {str(e)}")  # Debug line
        return f"Error generating PDF: {str(e)}"


# ----------------------------------------------------------
# Agent Initialization Function 
# ----------------------------------------------------------
def create_agent():
    return Agent(
        model=Gemini(id="gemini-2.5-flash-lite", api_key=st.session_state.GOOGLE_API_KEY),
        tools=[DuckDuckGoTools(), get_dicom_metadata, generate_pdf_report],
        markdown=True
    )
  
  
    
# ----------------------------------------------------------
# Prompt Templates
# ----------------------------------------------------------
def get_analysis_prompt(is_dicom: bool) -> str:
    base_prompt = "You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging.\n\n"
    if is_dicom:
        base_prompt += "IMPORTANT: First, call the get_dicom_metadata tool to retrieve patient information from this DICOM file.\n\n"
    base_prompt += """Analyze the patient's medical image and provide a comprehensive analysis with the following structure (use these EXACT headings with ### markdown format):

### 1. Image Type & Region
- Specify imaging modality
- Identify anatomical region and positioning
- Comment on image quality

### 2. Key Findings
- List primary observations systematically
- Note abnormalities with measurements, location, size, shape
- Rate severity: Normal/Mild/Moderate/Severe

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence
- List differential diagnoses
- Note critical/urgent findings

### 4. Patient-Friendly Explanation
- Explain findings in simple language
- Avoid jargon or define terms

### 5. Research Context
- Use DuckDuckGo to find recent medical literature and standard treatment protocols
- Provide 2-3 references with links

IMPORTANT FORMATTING RULES:
1. Use the exact section headings shown above (### 1. Image Type & Region, ### 2. Key Findings, etc.)
2. Write each section heading ONLY ONCE
3. Do NOT repeat the section structure multiple times
4. Format your response using clear markdown with bullet points under each section
5. Be concise yet thorough

After completing your analysis, call the generate_pdf_report tool with your complete analysis text to create a downloadable PDF report.
"""
    return base_prompt

def clean_analysis_text(text: str) -> str:
    """Clean up the analysis text to remove duplicates and ensure proper formatting."""
    import re
    
    # Remove any duplicate section headers that appear multiple times
    sections = [
        "### 1. Image Type & Region",
        "### 2. Key Findings", 
        "### 3. Diagnostic Assessment",
        "### 4. Patient-Friendly Explanation",
        "### 5. Research Context"
    ]
    
    # Find first occurrence of each section
    first_positions = {}
    for section in sections:
        matches = list(re.finditer(re.escape(section), text))
        if matches:
            first_positions[section] = matches[0].start()
    
    # If we found duplicate sections, rebuild the text
    if any(len(list(re.finditer(re.escape(s), text))) > 1 for s in sections):
        # Sort sections by their first appearance
        sorted_sections = sorted(first_positions.items(), key=lambda x: x[1])
        
        # Extract content between sections
        result_parts = []
        for i, (section, pos) in enumerate(sorted_sections):
            start = pos
            # Find the end (next section or end of text)
            if i < len(sorted_sections) - 1:
                end = sorted_sections[i + 1][1]
            else:
                end = len(text)
            
            # Extract this section's content (only first occurrence)
            section_content = text[start:end]
            # Remove any duplicate occurrences of the same section header within this content
            section_content = re.sub(
                f"({re.escape(section)}.*?){re.escape(section)}", 
                r"\1", 
                section_content, 
                flags=re.DOTALL
            )
            result_parts.append(section_content)
        
        text = "".join(result_parts)
    
    # Ensure proper markdown formatting
    text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
    
    return text.strip()

# ----------------------------------------------------------
# Main UI
# ----------------------------------------------------------
col_theme1, col_theme2 = st.columns([6, 1])
with col_theme2:
    theme_icon = "🌙" if st.session_state.theme == "light" else "☀️"
    if st.button(theme_icon, key="theme_toggle_main"):
        toggle_theme()
        st.rerun()

st.markdown("""
<div class="main-header">
    <h1>Medical Imaging AI</h1>
    <p>Advanced AI-Powered Radiology Analysis & Report Generation</p>
</div>
""", unsafe_allow_html=True)

# Supported Formats
st.markdown("### 📂 Supported Formats")
col_fmt1, col_fmt2, col_fmt3, col_fmt4 = st.columns(4)
with col_fmt1: st.info("**DICOM**\n\n.dcm, .dicom")
with col_fmt2: st.info("**PNG**\n\n.png")
with col_fmt3: st.info("**JPEG**\n\n.jpg, .jpeg")
with col_fmt4: st.info("**Features**\n\n✨ AI Analysis\n📊 Metadata\n📄 PDF Report")
st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Upload Section (Full Width)
st.markdown("### 📤 Upload Medical Image")

uploaded = st.file_uploader(
    "Drag and drop your medical image here or click to browse",
    type=["png","jpg","jpeg","dcm","dicom"],
    help="Supported formats: PNG, JPG, JPEG, DICOM (.dcm, .dicom)"
)

if uploaded is not None:
    if st.session_state.get("last_file") != uploaded.name:
        st.session_state.analysis = None
        st.session_state.pdf = None
        st.session_state.last_file = uploaded.name

    file_bytes = uploaded.getvalue()
    ext = uploaded.name.split(".")[-1].lower()
    metadata = None
    image = None

    if ext in ["dcm","dicom"]:
        metadata, full_tags = extract_dicom_metadata_helper(file_bytes)
        image = dicom_to_pil_helper(file_bytes)
        # Save to context
        _tool_context["dicom_bytes"] = file_bytes
        _tool_context["dicom_metadata"] = metadata 
    else:
        image = PILImage.open(BytesIO(file_bytes))
        _tool_context["dicom_bytes"] = None
        _tool_context["dicom_metadata"] = None

    _tool_context["image"] = image
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Two columns: Image + DICOM Metadata (if available)
    img_col, meta_col = st.columns([2, 1], gap="large")
    
    with img_col:
        st.image(image, caption=f"📸 {uploaded.name}", use_container_width=True)
        
        if not st.session_state.GOOGLE_API_KEY:
            st.warning("⚠️ Please configure your Google API Key in the sidebar to enable analysis.")
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔍 Analyze Image", use_container_width=True):
                with st.spinner("🤖 AI is analyzing your medical image..."):
                    try:
                        medical_agent = create_agent()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                            image.save(tmp.name)
                            agno_image = AgnoImage(filepath=tmp.name)
                            prompt = get_analysis_prompt(is_dicom=ext in ["dcm","dicom"])
                            result = medical_agent.run(prompt, images=[agno_image])
                            
                            # Clean and store analysis
                            cleaned_analysis = clean_analysis_text(result.content)
                            st.session_state.analysis = cleaned_analysis
                            
                            # Always generate PDF after successful analysis
                            pdf_buffer = generate_pdf_helper(
                                image,
                                cleaned_analysis,
                                metadata if ext in ["dcm","dicom"] else None
                            )
                            st.session_state.pdf = pdf_buffer
                            
                        st.success("✅ Analysis completed successfully!")
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        st.info("Please check your API key and try again.")
    
    with meta_col:
        if ext in ["dcm","dicom"] and metadata:
            st.markdown("### 📊 Patient Data")
            
            # 1. Summary Card (Visible by default)
            # We build the HTML once to prevent duplication
            metadata_html = '<div class="custom-card">'
            for k, v in metadata.items():
                metadata_html += f'<div class="metadata-card"><b>{k}:</b> {v}</div>'
            metadata_html += '</div>'
            
            # Render the Summary Card
            st.markdown(metadata_html, unsafe_allow_html=True)

            # 2. Full Metadata Expander (Hidden by default)
            with st.expander("📂 View Full DICOM Tags"):
                st.markdown("""
                <style>
                    .small-font { font-size: 12px !important; font-family: monospace; }
                </style>
                """, unsafe_allow_html=True)
                
                if full_tags:
                    for key, val in full_tags.items():
                        st.markdown(f"<div class='small-font'><b>{key}:</b> {val}</div>", unsafe_allow_html=True)
                        
    # Analysis Results (Full Width Below)
    if st.session_state.analysis:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### 📋 Analysis Results")
        
        # Use Streamlit's native container with custom CSS class
        st.markdown("""
        <style>
        .analysis-results-box {
            background-color: """ + ("#1E293B" if st.session_state.theme == "dark" else "#FFFFFF") + """ !important;
            border: 1px solid """ + ("#334155" if st.session_state.theme == "dark" else "#E5E7EB") + """ !important;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px """ + ("rgba(0, 0, 0, 0.4)" if st.session_state.theme == "dark" else "rgba(0, 0, 0, 0.08)") + """;
            margin-bottom: 1rem;
        }
        .analysis-results-box h3 {
            color: """ + ("#F1F5F9" if st.session_state.theme == "dark" else "#2C3E50") + """ !important;
            font-weight: 600 !important;
            margin-top: 1.5rem !important;
            margin-bottom: 0.75rem !important;
            font-size: 1.25rem !important;
        }
        .analysis-results-box p, .analysis-results-box li {
            color: """ + ("#F1F5F9" if st.session_state.theme == "dark" else "#2C3E50") + """ !important;
        }
        .analysis-results-box ul {
            margin-left: 1.5rem;
        }
        .analysis-results-box strong {
            color: """ + ("#F1F5F9" if st.session_state.theme == "dark" else "#2C3E50") + """ !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Convert markdown to HTML
        from markdown2 import markdown
        analysis_html_content = markdown(st.session_state.analysis, extras=[
            "tables", "fenced-code-blocks", "header-ids", "break-on-newline"
        ])
        
        # Render in a styled div
        st.markdown(f'<div class="analysis-results-box">{analysis_html_content}</div>', unsafe_allow_html=True)

        if st.session_state.pdf:
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                label="📥 Download PDF Report",
                data=st.session_state.pdf,
                file_name="Radiology_Report.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
        else:
            st.warning("⚠️ PDF was not generated by the agent. Try analyzing again.")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown(f"""
<div class="fixed-footer">
    ⚕️ Medical Imaging AI - Powered by Gemini AI<br>
    <span style="opacity:0.7;">For educational and research purposes only. Always consult qualified healthcare professionals.</span>
</div>
""", unsafe_allow_html=True)
