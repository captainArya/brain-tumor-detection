import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Gemini Model
model = genai.GenerativeModel("gemini-2.5-flash")


def generate_report(tumor_type, confidence, location, area):

    prompt = f"""
You are an expert neuro-oncology assistant.

MRI ANALYSIS REPORT

Tumor Type Identified: {tumor_type}

AI Confidence Score: {confidence:.2f}%

Tumor Location: {location}

Tumor Area Covered: {area:.2f}%

Generate a detailed and professional medical report.

Include:

1. Clinical Summary
2. Tumor Description
3. Tumor Location Analysis
4. Area Coverage Interpretation
5. Risk Assessment
6. Recommended Next Steps
7. Patient-Friendly Explanation

Instructions:

If Tumor Type is Glioma:
- Explain glioma characteristics
- Mention possible treatment approaches

If Tumor Type is Meningioma:
- Explain meningioma origin
- Mention prognosis

If Tumor Type is Pituitary:
- Explain pituitary gland involvement
- Mention possible hormonal effects

If Tumor Type is No Tumor:
- Clearly state no tumor detected
- Give general monitoring advice

Mention:
- Confidence Score
- Tumor Location
- Area Covered

Add a disclaimer:

"This report is AI-generated and should not be considered a final medical diagnosis. Please consult a qualified healthcare professional."
"""

    response = model.generate_content(prompt)

    return response.text