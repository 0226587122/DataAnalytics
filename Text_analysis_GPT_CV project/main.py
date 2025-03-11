import os
import pdfplumber
import docx
import requests  # Used for making API calls to Monica

# Set your Monica API key
MONICA_API_KEY = "sk-8thp5lUTlulWvf3gZp4_imBC0HrRf89iACcPCcy9hTJQVjghFa_bG-B4PdzsToXz3sjAMTqxcMOPkFxei4uMFamCn0N3"

def extract_text_from_pdf(filepath):
    """Extracts text from a PDF file."""
    import pdfplumber  # Ensure pdfplumber is installed
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_docx(file_path):
    """Extracts text from a DOCX file."""
    import docx  # Ensure python-docx is installed
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs]).strip()

def analyze_cv(cv_text):
    """
    Sends CV text to Monica for analysis.
    
    Args:
        cv_text (str): Extracted text from the CV file.
        
    Returns:
        str: Analysis results from Monica.
    """
    # Define the analysis prompt
    prompt = f"""
    You are an expert AI recruiter analyzing a candidate's CV in IT, software engineering, data analytics, and computer science fields. 

    1. Identify and categorize the candidate's experience into fields (e.g., Software Engineering, Lecturer, Business, Finance).
    2. Suggest the main two areas of the candidate's expertise and the most relevant job roles based on the experience.
    3. Provide recommendations for improving the CV in three bullet points.

    CV Text:
    {cv_text}
    """

    try:
        response = requests.post(
            url="https://api.monica.ai/analyze",  # Verify this endpoint
            headers={
                "Authorization": f"Bearer {MONICA_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "monica-latest",  # Replace with Monica's model identifier if needed
                "prompt": prompt
            },
            timeout=10  # Set a timeout to prevent hanging
        )

        # Handle HTTP errors
        response.raise_for_status()

        # Parse the response
        return response.json().get("analysis", "No analysis found in the response.")

    except requests.exceptions.Timeout:
        return "Error: The request to Monica's API timed out. Please try again later."
    except requests.exceptions.ConnectionError as e:
        return f"Connection error: Unable to reach the Monica API server. Details: {str(e)}"
    except requests.exceptions.HTTPError as e:
        return f"HTTP error: {str(e)} - {response.text}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def determine_file_type(file_path):
    """
    Determines the file type based on the extension and extracts text accordingly.
    
    Args:
        file_path (str): Path to the CV file.
        
    Returns:
        str: Extracted text from the CV file.
    """
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format! Please provide a PDF or DOCX file.")

if __name__ == "__main__":
    # Prompt user for CV file path
    file_path = input("Enter CV file path (PDF/DOCX): ").strip()

    # Check if file exists
    if not os.path.exists(file_path):
        print("File not found! Please check the file path and try again.")
        exit()

    try:
        # Extract text from the CV file
        cv_text = determine_file_type(file_path)

        print("\nAnalyzing CV with Monica...\n")
        analysis_result = analyze_cv(cv_text)

        print("\n--- CV Analysis Results ---\n")
        print(analysis_result)

    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
