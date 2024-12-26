import json
import os

import gradio as gr
import openai
import PyPDF2
import textract
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
#from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from pymongo import MongoClient

# Initialize OpenAI API key and MongoDB client
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
mongo_uri = os.getenv('mongo_uri')
model = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0)
client = MongoClient(mongo_uri)
db = client["forgeAI"]
collection = db["resume_parser_data"]

# Pydantic models and helper functions remain the same as in your original code
# [Previous SkillMatrix, NonSkillMatrix classes and other helper functions remain unchanged]
# Define Pydantic data models for structured output
class SkillMatrix(BaseModel):
    technical_skills: list = Field(description="list of technical skills with experience example python: '2 years' ")
    certifications: list = Field(description="list of certifications, each certification has information like issued date, expiry date, issued by organization")
    years_of_experience: str = Field(description="Total years of experience")
    experience_details: list = Field(description="list of job roles with companies and duration")
    projects: list = Field(description="list of projects with descriptions")
    education: list = Field(description="list of educational qualifications")

class NonSkillMatrix(BaseModel):
    candidate_name: str
    contact_number: str
    email_id: str
    skype_id: str
    visa_status: str
    visa_validity: str
    us_entry_date: str
    current_location: str
    relocate: str
    onsite: str
    available_for_interviews: str
    notice_period: str
    highest_education: str
    reason_for_change: str
    linkedin_id: str
    date_of_birth: str
    ssn_last_four_digits: int
    passport_number: str
    hotlist_template: dict = Field(description="Candidate initials and brief summary")

# Initialize Langchain's model
model = ChatOpenAI(temperature=0)

def extract_text(file):
    text = ""
    
    # Check if the file is a PDF
    if file.name.endswith(".pdf"):
        try:
            with open(file.name, "rb") as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        except Exception as e:
            text = f"Error reading PDF file: {str(e)}"
    
    # Check if the file is a DOC file
    elif file.name.endswith(".doc"):
        try:
            # Try to extract text from .doc file using textract
            text = textract.process(file.name, extension='doc').decode("utf-8")
        except Exception as e:
            text = f"Error reading DOC document with textract: {str(e)}"
    
    # Check if the file is a DOCX file, use textract with additional error handling
    elif file.name.endswith(".docx"):
        try:
            # Try to extract text from .doc file using textract
            text = textract.process(file.name).decode("utf-8")
        except Exception as e:
            text = f"Error reading DOCX document with textract: {str(e)}"
    
    # Check if the file is a TXT file
    elif file.name.endswith(".txt"):
        try:
            with open(file.name, "r", encoding="utf-8") as txt_file:
                text = txt_file.read()
        except Exception as e:
            text = f"Error reading TXT file: {str(e)}"
    
    # Handle unsupported file formats
    else:
        text = "Unsupported file format. Please upload a PDF, DOC, DOCX, or TXT file."
    
    print(text)
    return text
############################################

# For skill matrix
def generate_skill_matrix(extracted_text, model_structure):
    # Define the JSON output parser with the specified Pydantic model structure
    parser = JsonOutputParser(pydantic_object=model_structure)
    
    # Define the prompt template with embedded AI role and task instructions
    prompt_template = PromptTemplate(
        template="""
        AI ROLE:
        You are an AI assistant designed to parse and analyze candidate resumes, extracting specific information to create structured JSON data for recruitment purposes.

        TASK OBJECTIVE:
        Your task is to extract and categorize information from a given resume, focusing on Skill Data.

        TASK INPUT:
        You will receive a candidate's resume in plain text or JSON format, containing various sections such as personal details, skills, experience, education, certifications, and projects.

        TASK INSTRUCTIONS:
        1. Analyze the Resume: Thoroughly read the entire content of the resume to understand its structure and information.
        2. Extract Skill Data: Extract and categorize the following as "Skill Data":
           - Technical Skills List: Identify any programming languages, tools, software, or technical abilities mentioned, along with the number of years of experience for each skill.
           - Certifications and Their Details: Extract all certifications, along with any relevant details (e.g., name of certification, institution, issued date).
           - Years of Experience: Calculate or extract the total number of years of experience (including internships if relevant).
           - Experience Details: Extract previous job titles, companies, start year, and end year, including internships if mentioned.
           - Projects: List any projects, including names, descriptions, and roles.
           - Education Details: Extract educational qualifications, institutions, start year, end year, and any grades or honors mentioned.
        3. Avoid Hallucination: Only use the extracted data that is explicitly mentioned in the resume. Do not generate or infer information that is not present.
        4. Do not infer years of experience from the summary. Instead, calculate the total years of experience by analyzing each position listed in the resume. For each role, identify the relevant technologies used and attribute the corresponding number of years to each technology in the skill matrix. If that technology is used in multiple roles.

        OUTPUT FORMAT:
        Do not infer years of experience from the summary. Instead, calculate the total years of experience by analyzing each position listed in the resume. For each role, identify the relevant technologies used and attribute the corresponding number of years to each technology in the skill matrix. If that technology is used in multiple roles.
        Do not Fabricate the data with place holders if data not in resume like DD-MM-YYYY for date etc leave it blank
        Return only the modified JSON structure as plain JSON, without additional text, explanations, or code blocks. Do not enclose it in ```json ```. The JSON structure should follow the format below:

        {{
          "Skill Data": {{
            "Technical Skills": [
              {{
                "name": "Skill Name 1",
                "years_of_experience": "X years"
              }},
              {{
                "name": "Skill Name 2",
                "years_of_experience": "Y years"
              }}
              // More skills as needed
            ],
            "Certifications": [
              {{
                "name": "Certification Name 1",
                "institution": "Institution Name 1",
                "issued_date": "YYYY-MM-DD"
              }},
              {{
                "name": "Certification Name 2",
                "institution": "Institution Name 2",
                "issued_date": "YYYY-MM-DD"
              }}
              // More certifications as needed
            ],
            "Years of Experience": "X years",
            "Experience Details": [
              {{
                "role": "Job Title 1",
                "company": "Company Name 1",
                "start_year": "YYYY",
                "end_year": "YYYY"
              }},
              {{
                "role": "Job Title 2",
                "company": "Company Name 2",
                "start_year": "YYYY",
                "end_year": "YYYY"
              }}
              // More experiences as needed
            ],
            "Projects": [
              {{
                "name": "Project Name 1",
                "description": "Project Description 1",
                "role": "Role in Project 1"
              }},
              {{
                "name": "Project Name 2",
                "description": "Project Description 2",
                "role": "Role in Project 2"
              }}
              // More projects as needed
            ],
            "Education": [
              {{
                "degree": "Degree Name 1",
                "institution": "Institution Name 1",
                "start_year": "YYYY",
                "end_year": "YYYY",
                "grades": "Grades or honors (if available)"
              }},
              {{
                "degree": "Degree Name 2",
                "institution": "Institution Name 2",
                "start_year": "YYYY",
                "end_year": "YYYY",
                "grades": "Grades or honors (if available)"
              }}
              // More education entries as needed
            ]
          }}
        }}

        {format_instructions}

        Resume Text:
        {resume_text}
        """,
        input_variables=["resume_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    # Define the chain to process the template through the model and parse the response
    chain = prompt_template | model | parser

    # Invoke the chain with the resume text as input
    response = chain.invoke({"resume_text": extracted_text})
    return response

# Function to generate structured matrices using Langchain's output parser
def generate_non_skill_matrix(extracted_text, model_structure):
    # Set up parser with the model structure
    parser = JsonOutputParser(pydantic_object=model_structure)
    
    # Prompt template with detailed instructions
    prompt_template = PromptTemplate(
        template="""
        ### Skill Matrix Extraction
        You are an AI assistant that helps with extracting skill matrices from resumes.

        **Resume Text**:
        \"\"\"{resume_text}\"\"\"

        AI ROLE
        You are an AI assistant designed to parse and analyze candidate resumes, extracting specific information to create structured JSON data for recruitment purposes.

        TASK OBJECTIVE
        Your task is to extract and categorize information from a given resume, focusing on Non Skill Data and creating a Candidate Hotlist Template.

        TASK INPUT
        You will receive a candidate's resume in plain text or JSON format, containing various sections such as personal details, skills, experience, education, certifications, and projects.

        TASK INSTRUCTIONS
        Analyze the Resume: Thoroughly read the entire content of the resume to understand its structure and information.

        Extract Non Skill Data: Extract and categorize the following as "Non Skill Data":
        - Candidate_name: Name of the candidate.
        - Contact_number: Contact number of the candidate.
        - Email_id: Email ID of the candidate.
        - Skype_id: Skype ID of the candidate.
        - Visa_status: Type of visa the candidate is on.
        - Visa_validity: The date until the visa is valid.
        - US_entry_date: Date on which the candidate entered the United States.
        - Current_location: Current location of the candidate.
        - Relocate: Whether the candidate is willing to relocate (Yes/No/Willing to Negotiate).
        - Onsite: Whether the candidate is willing to work onsite (Yes/No/Willing to Negotiate).
        - Available_for_interviews: Whether the candidate is available for interviews (Yes/No).
        - Notice_period: The notice period of the candidate.
        - Highest_education_till_date: Highest form of education pursued or currently being pursued by the candidate.
        - Reason_for_change: Reason for change from previous employment.
        - LinkedIn_id: LinkedIn ID of the candidate.
        - Date_of_birth: Date of birth of the candidate.
        - SSN_last_four_digits: Last four digits of the candidate's Social Security Number.
        - Passport_number: Passport number of the candidate.

        For any field whose value is not present in the resume, leave it empty.

        Create Candidate Hotlist Template: Using the extracted data:
        - Candidate Initials: Generate initials from the candidate’s full name.
        - Brief Summary: If the resume contains a summary, extract and use it. If not, create a concise summary highlighting the candidate's key skills, experience, and qualifications.

        Avoid Hallucination: Only extract data that is explicitly mentioned in the resume. Do not generate or infer information that is not present.

        OUTPUT FORMAT
        Return only the JSON without any additional text or messages. The JSON structure should follow the format below:

        {{
          "Non Skill Data": {{
            "Candidate_name": "Name of the candidate",
            "Contact_number": "Contact number",
            "Email_id": "Email ID",
            "Skype_id": "Skype ID",
            "Visa_status": "Type of visa",
            "Visa_validity": "Visa validity date (if available)",
            "US_entry_date": "US entry date (if available)",
            "Current_location": "Current location of the candidate",
            "Relocate": "Yes/No/Willing to Negotiate",
            "Onsite": "Yes/No/Willing to Negotiate",
            "Available_for_interviews": "Yes/No",
            "Notice_period": "Notice period (if available)",
            "Highest_education_till_date": "Highest form of education",
            "Reason_for_change": "Reason for change from previous employment",
            "LinkedIn_id": "LinkedIn ID",
            "Date_of_birth": "Date of birth (if available)",
            "SSN_last_four_digits": "Last four digits of SSN",
            "Passport_number": "Passport number (if available)"
          }},
          "Candidate Hotlist Template": {{
            "Initials": "Candidate Initials",
            "Summary": "Brief summary of the resume"
          }}
        }}.\n{format_instructions}
        """,
        input_variables=["resume_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Chain prompt template, model, and parser
    chain = prompt_template | model | parser

    # Invoke the chain to generate the response
    response = chain.invoke({"resume_text": extracted_text})
    return response

def process_resume(uploaded_file):
    if uploaded_file is None:
        return (
            None,  # skill_matrix_state
            None,  # non_skill_matrix_state 
            None,  # matrix_display
            None,  # modified_matrix_display
            None,  # save_status_display
            None,  # resume_upload
            gr.Markdown("⚠️ Please upload a resume file before processing.", visible=True)  # error_message
        )
    resume_text = extract_text(uploaded_file)
    skill_matrix = generate_skill_matrix(resume_text, SkillMatrix)
    non_skill_matrix = generate_non_skill_matrix(resume_text, NonSkillMatrix)
    return (
        skill_matrix, 
        non_skill_matrix, 
        skill_matrix, 
        non_skill_matrix, 
        gr.update(visible=True), 
        None,
        gr.Markdown(visible=False)  # Hide error message when processing is successful
    )

# Modify matrix function using ChatOpenAI model instead of openai.Completion
def modify_matrix(skill_matrix, non_skill_matrix, matrix_type, user_input):
    # Select the matrix to modify based on user selection
    target_matrix = skill_matrix if matrix_type == "Skill Matrix" else non_skill_matrix

    # Construct the prompt based on the selected matrix type
    modification_prompt = (
        f"You are an assistant tasked with updating a JSON structure that represents a {matrix_type}.\n\n"
        
        "AI ROLE:\n"
        "You are a data processing assistant specializing in modifying structured JSON data for recruitment purposes.\n\n"
        
        "TASK INSTRUCTIONS:\n"
        "1. Analyze the provided JSON structure carefully to understand its format and values.\n"
        "2. Accurately interpret the user modification request and apply the changes to the JSON structure.\n"
        "3. Ensure that only the specified modifications are applied, retaining existing values and fields that are not part of the request.\n"
        "4. Provide only the updated JSON structure as output without any additional formatting or explanations.\n\n"
        
        "TASK INPUT:\n"
        f"Current {matrix_type} JSON:\n{json.dumps(target_matrix)}\n\n"
        f"User Modification Request: '{user_input}'\n\n"
        
        "EXPECTED OUTPUT:\n"
        "Return only the updated JSON structure in plain JSON format, with no additional text or code blocks. "
    )


    modification_response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": modification_prompt}
        ],
        temperature=0
    )
    
    try:
        updated_matrix = json.loads(modification_response.choices[0].message.content)
    except json.JSONDecodeError:
        updated_matrix = {"error": "Failed to apply modification. Please ensure the input format is correct."}
    if matrix_type == "Skill Matrix":
        skill_matrix = updated_matrix
    else:
        non_skill_matrix = updated_matrix
    print ("modified skill matrix json :", updated_matrix)
    return skill_matrix, non_skill_matrix, skill_matrix, non_skill_matrix, gr.update(selected="matrices_tab")

def save_to_mongo(skill_matrix, non_skill_matrix):
    try:
        document = {"Skill Matrix": skill_matrix, "Non-Skill Matrix": non_skill_matrix}
        document_id = collection.insert_one(document).inserted_id
        return {"status": "Success", "message": f"Matrices saved to MongoDB with ID: {document_id}"}, gr.update(visible=True)
    except Exception as e:
        return {"status": "Error", "message": str(e)}, gr.update(visible=True)

def update_matrix_view(matrix_type, skill_matrix, non_skill_matrix):
    if matrix_type == "Skill Matrix":
        return skill_matrix
    else:
        return non_skill_matrix

def navigate_to_modify_tab():
    return gr.update(selected="modify_tab")

def navigate_to_mongo_tab():
    return gr.update(selected="mongo_tab")

# Enhanced Gradio interface with tabs
with gr.Blocks(title="Resume Parser Dashboard") as iface:
    gr.Markdown("# Resume Parser Dashboard")
    
    # States for storing matrices
    skill_matrix_state = gr.State()
    non_skill_matrix_state = gr.State()
    
    with gr.Tabs() as tabs:
        # Tab 1: Resume Upload
        with gr.Tab("Upload Resume", id="upload_tab"):
            gr.Markdown("### Upload Your Resume")
            resume_upload = gr.File(label="Upload Resume (PDF, DOC, DOCX, or TXT)", type="filepath")
            error_message = gr.Markdown(visible=False)
            submit_button = gr.Button("Process Resume", variant="primary")
        
        # Tab 2: Matrices View
        with gr.Tab("View Matrices", id="matrices_tab"):
            gr.Markdown("### View Generated Matrices")
            matrix_type_view = gr.Dropdown(
                choices=["Skill Matrix", "Non-Skill Matrix"],
                label="Select Matrix Type",
                value="Skill Matrix"
            )
            matrix_display = gr.JSON(label="Matrix Content")
            
            with gr.Row():
                modify_nav_button = gr.Button("Need Changes")
                save_nav_button = gr.Button("Save to MongoDB")
        
        # Tab 3: Modify Matrices
        with gr.Tab("Modify Matrices", id="modify_tab"):
            gr.Markdown("### Modify Selected Matrix")
            matrix_type_modify = gr.Dropdown(
                choices=["Skill Matrix", "Non-Skill Matrix"],
                label="Select Matrix to Modify"
            )
            modification_input = gr.Textbox(
                label="Request Modifications",
                placeholder="Describe the changes you want to make..."
            )
            with gr.Row():
                save_nav_button = gr.Button("Save to MongoDB")
                apply_changes_button = gr.Button("Apply Changes", variant="primary")            
            modified_matrix_display = gr.JSON(label="Modified Matrix")
        
        # Tab 4: MongoDB Status
        with gr.Tab("Save Status", id="mongo_tab"):
            gr.Markdown("### MongoDB Save Status")
            save_status_display = gr.JSON(label="Save Status", visible=False)
    
    # Event handlers
    submit_button.click(
        fn=process_resume,
        inputs=[resume_upload],
        outputs=[
            skill_matrix_state,
            non_skill_matrix_state,
            matrix_display,
            modified_matrix_display,
            save_status_display,
            resume_upload,
            error_message
        ]
    ).then(
        fn=lambda skill_matrix: gr.update(selected="matrices_tab") if skill_matrix is not None else None,  # Only navigate if processing was successful
        inputs=[skill_matrix_state],
        outputs=tabs
    )
    
    matrix_type_view.change(
        fn=update_matrix_view,
        inputs=[matrix_type_view, skill_matrix_state, non_skill_matrix_state],
        outputs=matrix_display
    )
    
    modify_nav_button.click(
        fn=navigate_to_modify_tab,
        outputs=tabs
    )
    
    save_nav_button.click(
        fn=navigate_to_mongo_tab,
        outputs=tabs
    ).then(
        fn=save_to_mongo,
        inputs=[skill_matrix_state, non_skill_matrix_state],
        outputs=[save_status_display, save_status_display]
    )
    
    apply_changes_button.click(
        fn=modify_matrix,
        inputs=[
            skill_matrix_state,
            non_skill_matrix_state,
            matrix_type_modify,
            modification_input
        ],
        outputs=[
            matrix_display,
            modified_matrix_display,
            skill_matrix_state,
            non_skill_matrix_state
            
        ]
    )
    # Add this event handler after your other event handlers
    matrix_type_modify.change(
        fn=update_matrix_view,
        inputs=[matrix_type_modify, skill_matrix_state, non_skill_matrix_state],
        outputs=modified_matrix_display
    )
# Launch the interface
""" if __name__ == "__main__":
    try:
        iface.launch(server_port=7860,share=True)
    except RuntimeError:
        iface.launch() """