"""
Streamlit app: Medical Imaging Diagnosis Agent
Cleaned and optimized for Streamlit Cloud.

Key fixes applied:
- Uses st.secrets["GOOGLE_API_KEY"] for API key (do not store key in repo).
- Caches the Agent with @st.cache_resource so Gemini is initialized once.
- Adds safe_agent_run() with exponential backoff for 429 handling.
- Compresses/resizes images before sending to the agent to reduce payload.
- Builds a robust requirements hint in the file header.

Before deploying:
1. Add the following to Streamlit Cloud Secrets (Manage app → Secrets):
   GOOGLE_API_KEY = "your-google-api-key-here"

2. Add a requirements.txt to your repo with at least:
   streamlit
   pydicom
   numpy
   Pillow
   markdown2
   duckduckgo-search
   ddgs
   google-genai
   google-generativeai
   agno
   opencv-python-headless
   scikit-image
   reportlab

Replace/adjust package versions as needed.

"""

import os
import time
import tempfile
from io import BytesIO
from typing import Optional

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

# Show error details in Streamlit (useful for debugging)
st.set_option("client.showErrorDetails", True)

# ------------------------------------------------------------------
# Utility: Safe agent run with retries/backoff
# ------------------------------------------------------------------

def safe_agent_run(agent, prompt, images=None, retries=4, backoff_factor=2):
    """Run agent.run with exponential backoff on 429 or rate-limit errors.

    Returns the agent output (or raises the last exception after retries).
    """
    last_exception = None
    for attempt in range(retries):
        try:
            if images:
                return agent.run(prompt, images=images)
            return agent.run(prompt)
        except Exception as e:
            last_exception = e
            msg = str(e).lower()
            # If error looks like rate-limit (429) or too many requests -> retry
            if "429" in msg or "too many requests" in msg or "rate limit" in msg:
                sleep_time = backoff_factor ** attempt
                st.warning(f"Rate limit detected. Retrying in {sleep_time} seconds (attempt {attempt + 1}/{retries})...")
                time.sleep(sleep_time)
                continue
            # For auth errors, bail out early
            if "unauthorized" in msg or "invalid" in msg or "api key" in msg:
                raise
            # Otherwise, not retryable
            raise
    # all retries exhausted
    raise last_exception


# ------------------------------------------------------------------
# DICOM / image helpers
# ------------------------------------------------------------------

def extract_dicom_metadata_helper(file_bytes: bytes) -> dict:
    """Extract simple DICOM metadata from bytes"""
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


def dicom_to_pil_helper(file_bytes: bytes, fallback_size=(512, 512)) -> PILImage:
    """Convert DICOM bytes to a PIL Image (grayscale). Returns fallback image on failure."""
    try:
        ds = pydicom.dcmread(BytesIO(file_bytes), force=True)
        data = apply_voi_lut(ds.pixel_array, ds)

        # handle multi-frame
        if data.ndim == 3:
            data = data[0]

        # MONOCHROME1 means inverted
        if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
            data = np.max(data) - data

        # normalize
        data = data.astype(np.float32)
        data = (data - data.min()) / (data.max() - data.min() + 1e-9) * 255.0
        data = np.clip(data, 0, 255).astype(np.uint8)

        img = PILImage.fromarray(data).convert("L")
        # ensure reasonable size
        img = img.resize(fallback_size)
        return img
    except Exception:
        return PILImage.new("L", fallback_size, color=128)


# ------------------------------------------------------------------
# PDF generation helper (ReportLab)
# ------------------------------------------------------------------

def generate_pdf_helper(image: PILImage, analysis_text: str, dicom_metadata: Optional[dict] = None) -> BytesIO:
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Heading1Custom", parent=styles["Heading1"], fontSize=18, spaceAfter=12))
    styles.add(ParagraphStyle(name="Heading2Custom", parent=styles["Heading2"], fontSize=15, spaceAfter=10))
    styles.add(ParagraphStyle(name="NormalCustom", parent=styles["Normal"], fontSize=11, leading=15))

    story = []

    story.append(Paragraph("<b>Radiology Report</b>", styles["Heading1Custom"]))
    story.append(Spacer(1, 12))

    if dicom_metadata:
        story.append(Paragraph("<b>👤 Patient Information</b>", styles["Heading2Custom"]))
        for key, value in dicom_metadata.items():
            story.append(Paragraph(f"<b>{key}:</b> {value}", styles["NormalCustom"]))
            story.append(Spacer(1, 4))
        story.append(Spacer(1, 12))

    # add image
    img_width = 250
    aspect_ratio = image.height / max(image.width, 1)
    img_height = int(img_width * aspect_ratio)
    tmp_buffer = BytesIO()
    image.resize((img_width, img_height)).save(tmp_buffer, format="PNG")
    tmp_buffer.seek(0)

    story.append(RLImage(tmp_buffer))
    story.append(Spacer(1, 20))

    # markdown -> basic HTML -> paragraphs
    html_text = markdown(analysis_text or "", extras=[
        "tables",
        "fenced-code-blocks",
        "strike",
        "underline",
        "footnotes",
        "cuddled-lists",
    ])

    for line in html_text.split("\n"):
        text = line.strip()
        if not text:
            continue
        if text.startswith("<h1>"):
            content = text.replace("<h1>", "").replace("</h1>", "")
            story.append(Paragraph(content, styles["Heading1Custom"]))
        elif text.startswith("<h2>"):
            content = text.replace("<h2>", "").replace("</h2>", "")
            story.append(Paragraph(content, styles["Heading2Custom"]))
        else:
            story.append(Paragraph(text, styles["NormalCustom"]))
        story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ------------------------------------------------------------------
# Agent Tools / Context
# ------------------------------------------------------------------
_tool_context = {
    "dicom_bytes": None,
    "image": None,
    "dicom_metadata": None,
}


def get_dicom_metadata() -> str:
    if _tool_context["dicom_bytes"] is None:
        return "No DICOM file available. This image is not in DICOM format."
    metadata = extract_dicom_metadata_helper(_tool_context["dicom_bytes"])
    result = "DICOM Metadata:\n\n"
    for key, value in metadata.items():
        result += f"• {key}: {value}\n"
    return result


def generate_pdf_report(analysis_text: str) -> str:
    if _tool_context["image"] is None:
        return "Error: No image available for PDF generation."
    try:
        pdf_buffer = generate_pdf_helper(
            _tool_context["image"],
            analysis_text,
            _tool_context.get("dicom_metadata"),
        )
        # store pdf for download
        st.session_state.pdf = pdf_buffer
        return "PDF report successfully generated! The user can now download it."
    except Exception as e:
        return f"Error generating PDF: {str(e)}"


# ------------------------------------------------------------------
# Cached Agent (single initialization)
# ------------------------------------------------------------------
@st.cache_resource
def get_cached_agent():
    """Create and return the Agent. This is cached so the model isn't recreated on each rerun."""
    # Read key from Streamlit secrets (recommended for Streamlit Cloud)
    if "GOOGLE_API_KEY" not in st.secrets:
        raise RuntimeError("GOOGLE_API_KEY not found in Streamlit secrets. Add it via Manage app -> Secrets.")

    api_key = st.secrets["GOOGLE_API_KEY"]

    return Agent(
        model=Gemini(id="gemini-2.5-pro", api_key=api_key),
        tools=[DuckDuckGoTools(), get_dicom_metadata, generate_pdf_report],
        markdown=True,
    )


# ------------------------------------------------------------------
# Prompt template
# ------------------------------------------------------------------

def get_analysis_prompt(is_dicom: bool) -> str:
    base_prompt = (
        "You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging.\n\n"
    )
    if is_dicom:
        base_prompt += (
            "IMPORTANT: First, call the get_dicom_metadata tool to retrieve patient information from this DICOM file.\n\n"
        )
    base_prompt += (
        "Analyze the patient's medical image and structure your response as follows:\n\n"
        "### 1. Image Type & Region\n- Specify imaging modality\n- Identify anatomical region and positioning\n- Comment on image quality\n\n"
        "### 2. Key Findings\n- List primary observations systematically\n- Note abnormalities with measurements, location, size, shape\n- Rate severity: Normal/Mild/Moderate/Severe\n\n"
        "### 3. Diagnostic Assessment\n- Provide primary diagnosis with confidence\n- List differential diagnoses\n- Note critical/urgent findings\n\n"
        "### 4. Patient-Friendly Explanation\n- Explain findings in simple language\n- Avoid jargon or define terms\n\n"
        "### 5. Research Context\n- Use DuckDuckGo to find recent medical literature and standard treatment protocols\n- Provide 2-3 references with links\n\n"
        "After completing your analysis, MUST call the generate_pdf_report tool with your complete analysis text to create a downloadable PDF report for the patient.\n\n"
        "Format your response using clear markdown headers and bullet points. Be concise yet thorough."
    )
    return base_prompt


# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------

st.title("🏥 Medical Imaging Diagnosis Agent (Cloud-ready)")
st.caption("AI-powered radiology analysis with automated report generation")

if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "pdf" not in st.session_state:
    st.session_state.pdf = None

# File uploader
uploaded = st.file_uploader("Upload Medical Image", ["png", "jpg", "jpeg", "dcm", "dicom"])

if uploaded is None:
    st.info("Please upload a medical image to begin analysis.")
    with st.expander("ℹ️ Supported Formats"):
        st.markdown(
            """
        - **DICOM** (.dcm, .dicom) - Medical imaging standard format
        - **Images** (.png, .jpg, .jpeg) - Standard image formats

        The agent will automatically:
        - Extract DICOM metadata (if applicable)
        - Perform medical image analysis
        - Generate a professional PDF report
        """
        )

else:
    # Reset state on new file
    if st.session_state.get("last_file") != uploaded.name:
        st.session_state.analysis = None
        st.session_state.pdf = None
        st.session_state.last_file = uploaded.name

    file_bytes = uploaded.getvalue()
    ext = uploaded.name.split(".")[-1].lower()

    # Process
    metadata = None
    image = None

    if ext in ["dcm", "dicom"]:
        metadata = extract_dicom_metadata_helper(file_bytes)
        image = dicom_to_pil_helper(file_bytes)
        _tool_context["dicom_bytes"] = file_bytes
        _tool_context["dicom_metadata"] = metadata
        with st.expander("DICOM Metadata", expanded=False):
            for k, v in metadata.items():
                st.write(f"**{k}:** {v}")
    else:
        # load normal image and resize to reasonable size
        image = PILImage.open(BytesIO(file_bytes)).convert("RGB")
        image = image.resize((512, 512))
        _tool_context["dicom_bytes"] = None
        _tool_context["dicom_metadata"] = None

    _tool_context["image"] = image

    st.image(image, caption="Uploaded Image", width=400)

    # Analysis button
    if st.button("Analyze Image", type="primary"):
        # Ensure API key in secrets
        if "GOOGLE_API_KEY" not in st.secrets:
            st.error("GOOGLE_API_KEY not found. Please add it via Manage app -> Secrets.")
        else:
            with st.spinner("Agent is analyzing the image..."):
                try:
                    # get cached agent (will not reinitialize on reruns)
                    medical_agent = get_cached_agent()

                    # Save image to temp file for AgnoImage
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        # compress image before sending
                        send_img = image.copy().convert("RGB")
                        send_img.thumbnail((512, 512))
                        send_img.save(tmp.name, format="PNG", optimize=True)

                        agno_image = AgnoImage(filepath=tmp.name)

                        # Prompt
                        prompt = get_analysis_prompt(is_dicom=ext in ["dcm", "dicom"])

                        # Run agent with retry/backoff
                        output = safe_agent_run(medical_agent, prompt, images=[agno_image])

                        # Agno Agent run may return an object – attempt to get content
                        # We support multiple possible output shapes
                        content = None
                        if hasattr(output, "content"):
                            content = output.content
                        elif isinstance(output, dict) and "content" in output:
                            content = output["content"]
                        else:
                            content = str(output)

                        st.session_state.analysis = content

                    st.success("Analysis completed successfully!")

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.info("Check your API key, billing, and that you are not exceeding rate limits.")

    # Display results + PDF download
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
            )
        else:
            st.warning("PDF was not generated by the agent. Try analyzing again.")

# End of app
