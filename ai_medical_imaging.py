import os
import tempfile
from io import BytesIO
from PIL import Image as PILImage
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
from markdown2 import markdown
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from typing import Optional


# ----------------------------------------------------------
# Sidebar API Key
# ----------------------------------------------------------
if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state.GOOGLE_API_KEY = None


with st.sidebar:
    st.title("ℹ️ Configuration")

    if not st.session_state.GOOGLE_API_KEY:
        api_key = st.text_input("Enter your Google API Key:", type="password")
        st.caption("Get your API key from Google AI Studio.")
        if api_key:
            st.session_state.GOOGLE_API_KEY = api_key
            st.success("API Key Saved!")
            st.rerun()
    else:
        st.success("API Key is configured")
        if st.button("Reset API Key"):
            st.session_state.GOOGLE_API_KEY = None
            st.rerun()
        
        

# ----------------------------------------------------------
# Helper Functions (used by both tools and UI)
# ----------------------------------------------------------
def extract_dicom_metadata_helper(file_bytes):
    """Extract DICOM metadata from bytes"""
    try:
        ds = pydicom.dcmread(BytesIO(file_bytes), force=True)
        rows = getattr(ds, "Rows", "N/A")
        cols = getattr(ds, "Columns", "N/A")

        return {
           "Patient Name": str(getattr(ds, "PatientName", "N/A")),
           "Patient ID": str(getattr(ds, "PatientID", "N/A")),
           "Patient Sex": str(getattr(ds, "PatientSex", "N/A")),
           "Patient Age": str(getattr(ds, "PatientAge", "N/A")),
           "Patient Weight": str(getattr(ds, "PatientWeight", "N/A")),
           "Body Part Examined": str(getattr(ds, "BodyPartExamined", "N/A")),
           "Modality": str(getattr(ds, "Modality", "N/A")),
           "Study Description": str(getattr(ds, "StudyDescription", "N/A")),
           "Image Dimensions": f"{rows} x {cols}",
        }
    except Exception as e:
        return {"Error": str(e)}


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


def pet_ct_fusion(ct_bytes: bytes, pet_bytes: bytes, alpha=0.4):
    """
    Fuse CT (grayscale) with PET (heatmap SUV)
    """
    ct_ds = pydicom.dcmread(BytesIO(ct_bytes), force=True)
    pet_ds = pydicom.dcmread(BytesIO(pet_bytes), force=True)

    # --- CT IMAGE ---
    ct_img = apply_voi_lut(ct_ds.pixel_array, ct_ds).astype(float)
    ct_img = (ct_img - ct_img.min()) / (ct_img.max() - ct_img.min())
    ct_img = (ct_img * 255).astype(np.uint8)
    ct_img = PILImage.fromarray(ct_img).convert("RGB")

    # --- PET IMAGE (SUV) ---
    pet = pet_ds.pixel_array.astype(float)

    # SUV calculation (simplified standard)
    if hasattr(pet_ds, "RescaleSlope"):
        pet *= pet_ds.RescaleSlope
    if hasattr(pet_ds, "RescaleIntercept"):
        pet += pet_ds.RescaleIntercept

    pet = np.clip(pet, 0, np.percentile(pet, 99))
    pet = pet / pet.max()
    pet = (pet * 255).astype(np.uint8)

    pet_img = PILImage.fromarray(pet).resize(ct_img.size)
    pet_img = pet_img.convert("RGB")

    # --- FUSION ---
    fused = PILImage.blend(ct_img, pet_img, alpha=alpha)
    return fused



def generate_pdf_helper(image: PILImage, analysis_text: str, dicom_metadata: dict = None):
    """Generate PDF from image and analysis text"""
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Heading1Custom', parent=styles['Heading1'], fontSize=18, spaceAfter=12))
    styles.add(ParagraphStyle(name='Heading2Custom', parent=styles['Heading2'], fontSize=15, spaceAfter=10))
    styles.add(ParagraphStyle(name='NormalCustom', parent=styles['Normal'], fontSize=11, leading=15))

    story = []

    # Title
    story.append(Paragraph("<b>Radiology Report</b>", styles['Heading1Custom']))
    story.append(Spacer(1, 12))

    # Add Patient DICOM Metadata if available
    if dicom_metadata:
        story.append(Paragraph("<b>👤 Patient Information</b>", styles['Heading2Custom']))
        for key, value in dicom_metadata.items():
            story.append(Paragraph(f"<b>{key}:</b> {value}", styles['NormalCustom']))
            story.append(Spacer(1, 4))
        story.append(Spacer(1, 12))

    # Add image
    img_width = 250
    aspect_ratio = image.height / image.width
    img_height = int(img_width * aspect_ratio)
    tmp_buffer = BytesIO()
    image.resize((img_width, img_height)).save(tmp_buffer, format="PNG")
    tmp_buffer.seek(0)

    story.append(RLImage(tmp_buffer))
    story.append(Spacer(1, 20))

    # Convert markdown to HTML
    html_text = markdown(analysis_text, extras=[
        "tables",
        "fenced-code-blocks",
        "strike",
        "underline",
        "footnotes",
        "cuddled-lists"
    ])

    # HTML → PDF
    for line in html_text.split("\n"):
        if line.strip():
            if "<h1>" in line:
                story.append(Paragraph(line.replace("<h1>", "").replace("</h1>", ""), styles['Heading1Custom']))
            elif "<h2>" in line:
                story.append(Paragraph(line.replace("<h2>", "").replace("</h2>", ""), styles['Heading2Custom']))
            elif "<h3>" in line:
                story.append(Paragraph(line.replace("<h3>", "").replace("</h3>", ""), styles['Heading2Custom']))
            else:
                story.append(Paragraph(line, styles['NormalCustom']))
            story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ----------------------------------------------------------
# Agent Tool Functions (callable by the agent)
# ----------------------------------------------------------
# Store context globally for tool access
_tool_context = {
    "dicom_bytes": None,
    "image": None,
    "dicom_metadata": None
}

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


def generate_pdf_report(analysis_text: str) -> str:
    """
    Generate a professional PDF radiology report from your analysis.
    Call this tool with your complete markdown-formatted analysis after you finish the medical image analysis.
    
    Args:
        analysis_text: Your complete medical analysis in markdown format
        
    Returns:
        Confirmation message that PDF has been generated
    """
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
        
        return "PDF report successfully generated! The user can now download it."
    
    except Exception as e:
        return f"Error generating PDF: {str(e)}"


# ----------------------------------------------------------
# Agent Initialization Function
# ----------------------------------------------------------
def create_agent():
    """Create medical agent with tools"""
    return Agent(
        model=Gemini(id="gemini-2.5-pro", api_key=st.session_state.GOOGLE_API_KEY),
        tools=[
            DuckDuckGoTools(),
            get_dicom_metadata,
            generate_pdf_report
        ],
        markdown=True
    )


# ----------------------------------------------------------
# Prompt Templates
# ----------------------------------------------------------
def get_analysis_prompt(is_dicom: bool) -> str:
    """Generate appropriate prompt based on file type"""
    
    base_prompt = """You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging.

IMPORTANT RULES:
- You MUST be explicit about uncertainty.
- Assign confidence scores (%) to findings and diagnoses.
- If unsure, state low confidence clearly.

""" 
    
    if is_dicom:
        base_prompt += """IMPORTANT: First, call the get_dicom_metadata tool to retrieve patient information from this DICOM file.

"""
    
    base_prompt += """Analyze the patient's medical image and structure your response as follows:

### 1. Image Type & Region
- Specify imaging modality
- Identify anatomical region and positioning
- Comment on image quality

### 2. Key Findings 
- List primary observations systematically
- Note abnormalities with measurements, location, size, shape
- Rate severity: Normal/Mild/Moderate/Severe

### 3. Diagnostic Assessment
- Provide primary diagnosis with **confidence score**
- List differential diagnoses (with confidence)
- Note critical/urgent findings

### 4. Patient-Friendly Explanation
- Explain findings in simple language
- Avoid jargon or define terms

### 5. Research Context
- Use DuckDuckGo to find recent medical literature and standard treatment protocols
- Provide 2-3 references with links

After completing your analysis, MUST call the generate_pdf_report tool with your complete analysis text to create a downloadable PDF report for the patient.

Format your response using clear markdown headers and bullet points. Be concise yet thorough.
"""
    
    return base_prompt


# ----------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------
st.title("🏥 Medical Imaging Diagnosis Agent")
st.caption("AI-powered radiology analysis with automated report generation")

if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "pdf" not in st.session_state:
    st.session_state.pdf = None

uploaded = st.file_uploader("Upload Medical Image (CT / PET / PET-CT)", ["png", "jpg", "jpeg", "dcm", "dicom"])

# ------------ SHOW MESSAGE ONLY IF NO IMAGE ------------
if uploaded is None:
    st.info("Please upload a medical image to begin analysis.")
    
    with st.expander("ℹ️ Supported Formats"):
        st.markdown("""
        - **DICOM** (.dcm, .dicom) - Medical imaging standard format
        - **Images** (.png, .jpg, .jpeg) - Standard image formats
        
        The agent will automatically:
        - Extract DICOM metadata (if applicable)
        - Perform medical image analysis
        - Generate a professional PDF report
        """)
else:
    # ---------- Reset state if image changes ----------
    if st.session_state.get("last_file") != uploaded.name:
        st.session_state.analysis = None
        st.session_state.pdf = None
        st.session_state.last_file = uploaded.name

    file_bytes = uploaded.getvalue()
    ext = uploaded.name.split(".")[-1].lower()

    # Process image based on type
    metadata = None
    image = None
    
    if ext in ["dcm", "dicom"]:
        metadata = extract_dicom_metadata_helper(file_bytes)
        image = dicom_to_pil_helper(file_bytes)
        
        # Update tool context for DICOM
        _tool_context["dicom_bytes"] = file_bytes
        _tool_context["dicom_metadata"] = metadata
        
        # Display metadata in UI
        with st.expander("DICOM Metadata", expanded=False):
            for k, v in metadata.items():
                st.write(f"**{k}:** {v}")
    else:
        image = PILImage.open(BytesIO(file_bytes))
        _tool_context["dicom_bytes"] = None
        _tool_context["dicom_metadata"] = None
    
    # Update tool context with image
    _tool_context["image"] = image
    
    # Display image
    st.image(image, caption="Uploaded Image", width=400)

    # Analysis button
    if st.button("Analyze Image", type="primary") and st.session_state.GOOGLE_API_KEY:
        with st.spinner("Agent is analyzing the image..."):
            try:
                # Create agent
                medical_agent = create_agent()
                
                # Save image to temp file for agent
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    image.save(tmp.name)
                    agno_image = AgnoImage(filepath=tmp.name)

                    # Get prompt
                    prompt = get_analysis_prompt(is_dicom=ext in ["dcm", "dicom"])
                    
                    # Run agent
                    result = medical_agent.run(prompt, images=[agno_image])
                    st.session_state.analysis = result.content
                    
                st.success("Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.info("Please check your API key and try again.")

    # Display results
    if st.session_state.analysis:
        st.markdown("---")
        st.markdown("### 📋 Analysis Results")
        st.markdown(st.session_state.analysis)
       

        if st.session_state.pdf:
            st.markdown("---")
            st.download_button(
                "Download Report",
                st.session_state.pdf,
                file_name="Radiology_Report.pdf",
                mime="application/pdf",
                type="primary"
            )
        else:
            st.warning("PDF was not generated by the agent. Try analyzing again.")
    
    if not st.session_state.GOOGLE_API_KEY:
        st.warning("Please configure your Google API Key in the sidebar to enable analysis.")

