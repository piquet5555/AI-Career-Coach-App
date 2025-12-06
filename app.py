# ==============================================================================
# 2. PERMANENT FIX: Dependencies are installed externally (Cell 1)
# ==============================================================================

# --- IMPORTS ---
import streamlit as st
import os
import subprocess
import tempfile

# File processing imports
import pypdf
from docx import Document

# Imports for CrewAI and LangChain components
from crewai import Agent, Task, Crew, Process
from crewai_tools import RagTool, BaseTool # Removed SerperDevTool as it's not strictly needed here
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from textwrap import dedent
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup

# --- 1. CONFIGURATION AND TOOL DEFINITIONS ---

# LLM Setup (Technical Accomplishment: Multiple Models)
llm_fast = ChatOpenAI(temperature=0.0, model=os.environ.get("GPT3_MODEL_NAME", "gpt-3.5-turbo"))
llm_quality = ChatOpenAI(temperature=0.7, model=os.environ.get("OPENAI_MODEL_NAME", "gpt-4o"))

# --- Existing Custom Tool (Course Catalog) ---
class CourseCatalogSchema(BaseModel):
    course_name: str = Field(description="The specific course name to look up.")

class CourseCatalogTool(BaseTool):
    name: str = "Course Catalog Search"
    description: str = "Tool for looking up real-time availability and descriptions for suggested courses (e.g., 'CS 400')."
    args_schema = CourseCatalogSchema

    def _run(self, course_name: str) -> str:
        # This simulates an API call to a University Course Catalog
        if "CS 400" in course_name:
            return "CS 400 (Advanced DS&A) is available next semester (Spring 2026). Prerequisite: CS 320. Instructor: Prof. Lee."
        elif "IT 350" in course_name:
            return "IT 350 (Data Visualization) is available this semester (Fall 2025). Section is currently 80% full. Prerequisite: BA 205."
        else:
            return f"Course {course_name} not found in the current catalog."

course_api_tool = CourseCatalogTool()


# --- Resume Reading Tool ---
class ResumeReaderTool(BaseTool):
    name: str = "Resume Content Reader"
    description: str = "Tool for securely reading text content from an uploaded Streamlit file (PDF or DOCX)."

    def _run(self, file_path: str) -> str:
        """Reads content from a local file path."""
        file_extension = os.path.splitext(file_path)[1].lower()
        content = ""

        try:
            if file_extension == '.pdf':
                reader = pypdf.PdfReader(file_path)
                for page in reader.pages:
                    content += page.extract_text()
            elif file_extension == '.docx':
                doc = Document(file_path)
                for para in doc.paragraphs:
                    content += para.text + '\n'
            else:
                return f"Error: Unsupported file type: {file_extension}. Must be PDF or DOCX."

            return content[:8000] # Limit size to avoid excessive token usage
        except Exception as e:
            return f"Error reading file {file_path}: {e}"

resume_reader_tool = ResumeReaderTool()


# --- Job Description Scraper Tool ---
class JobDescriptionScraper(BaseTool):
    name: str = "Job Description Scraper"
    description: str = "Tool for fetching and cleaning the text content of a job posting URL."

    def _run(self, url: str) -> str:
        """Fetches and cleans text from a given URL."""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status() # Raise exception for bad status codes

            soup = BeautifulSoup(response.content, 'html.parser')

            # Simple cleaning: remove script and style tags
            for script_or_style in soup(['script', 'style']):
                script_or_style.decompose()

            # Get text and clean up whitespace
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            return text[:10000] # Limit size for performance and token management
        except Exception as e:
            return f"Error fetching or parsing URL {url}: {e}"

job_scraper_tool = JobDescriptionScraper()


# --- RAG KNOWLEDGE BASE SETUP ---
coach_knowledge_text = """
Job Competency: Software Engineer (Junior)
    - Required: Python (3+ yrs), SQL (databases), Git (version control).
    - Desired: Cloud Computing (AWS/Azure), Agile Methodology, Data Structures/Algorithms (DS&A).
    - Relevant Courses: CS 101 (Intro to Python), CS 320 (Databases), CS 400 (Advanced DS&A).
    - Interview Tips: Use the STAR method. Focus on problem-solving process, not just the answer.

Job Competency: Data Analyst (Entry):
    - Required: Excel (Advanced), SQL (data retrieval), Data Visualization (Tableau/PowerBI).
    - Desired: Python (Pandas), Statistical Modeling.
    - Relevant Courses: BA 205 (Statistics), IT 350 (Data Visualization).
    - Interview Tips: Be ready to explain your process for cleaning messy data.
"""

coach_rag_tool = RagTool(
    name="Career_Knowledge_Retriever",
    description="Tool for retrieving specific, cited information on job requirements, courses, and interview tips from the internal coach knowledge base.",
    content=coach_knowledge_text
)


# --- 2. AGENT DEFINITIONS ---

# Agent 1: The Recommender Agent (Uses fast LLM)
recommender_agent = Agent(
    role='Career Path Recommender',
    goal=f'Analyze a student\'s profile and recommend the single best-fit job role from the knowledge base, providing a clear explanation.',
    backstory='You are an expert HR professional who matches student skills, academics, and interests to career paths using data-driven methods.',
    tools=[coach_rag_tool],
    llm=llm_fast,
    verbose=True,
    allow_delegation=False
)

# Agent 2: The Preparer Agent (Uses quality LLM, can call external API)
preparer_agent = Agent(
    role='Job Preparation Guide',
    goal='Develop a personalized, detailed action plan for the recommended job role, including specific courses, skills, and interview steps, citing the source.',
    backstory='You are a certified career coach focused on creating actionable, step-by-step guidance plans for students to successfully secure a job.',
    tools=[coach_rag_tool, course_api_tool],
    llm=llm_quality,
    verbose=True,
    allow_delegation=False
)

# Agent 3: The Conversational Coach (Uses quality LLM, for follow-up chat)
chat_agent = Agent(
    role='Interactive Career Coach',
    goal='Provide conversational, supportive, and informative follow-up answers based on the student\'s previous plan and new questions.',
    backstory='You are a friendly coach with access to all previous conversations and preparation plans. You answer concisely and helpfully.',
    tools=[coach_rag_tool, course_api_tool],
    llm=llm_quality,
    verbose=False,
    allow_delegation=False
)

# Agent 4: The Critique Agent (Uses quality LLM, uses the new reader tool)
critique_agent = Agent(
    role='Professional Resume Reviewer',
    goal='Provide a comprehensive, section-by-section critique and specific actionable feedback for an uploaded resume.',
    backstory='You are a seasoned hiring manager with 10+ years of experience. You evaluate resumes on clarity, impact, relevance, and presentation.',
    tools=[resume_reader_tool, coach_rag_tool],
    llm=llm_quality,
    verbose=True,
    allow_delegation=False
)

# Agent 5: The Revision Agent (Uses quality LLM, uses reader and scraper tools)
revision_agent = Agent(
    role='Job-Targeted Resume Editor',
    goal='Analyze a resume against a specific job description and output a revised resume section that aligns perfectly with the job post.',
    backstory='You are a master of applicant tracking systems (ATS). Your job is to maximize keyword alignment and professional impact for a single target job.',
    tools=[resume_reader_tool, job_scraper_tool],
    llm=llm_quality,
    verbose=True,
    allow_delegation=False
)


# --- 3. STREAMLIT FRONTEND AND WORKFLOW ---
st.set_page_config(layout="wide")
st.title("🤖 AI-Powered Career Coach App")
st.markdown("Your personalized guide to the right career, and how to perfect your resume.")

# Initialize session state for multi-turn chat and storing the plan
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "plan_generated" not in st.session_state:
    st.session_state["plan_generated"] = False
if "resume_uploaded" not in st.session_state:
    st.session_state["resume_uploaded"] = None
if "critique_result" not in st.session_state:
    st.session_state["critique_result"] = None

# Pre-instantiate the chat crew (Performance optimization)
chat_crew_fixed = Crew(
    agents=[chat_agent],
    tasks=[],
    process=Process.sequential,
    verbose=0
)

# --- TAB INTERFACE ---
tab1, tab2, tab3 = st.tabs(["Career Plan Generator", "Resume Review & Critique", "Chat with Coach"])

with tab1:
    st.header("1. Career Plan Generator")
    st.markdown("Enter your data to receive a personalized job recommendation and preparation plan.")

    with st.form("career_coach_form"):
        student_name = st.text_input("Name", value="", placeholder="e.g., Jane Doe")
        academics = st.text_area("Academic Performance/Major", value="", placeholder="e.g., Computer Science Major, 3.8 GPA, relevant certifications")
        courses = st.text_area("Relevant Courses/Skills", value="", placeholder="List all relevant programming languages, software, and tools (e.g., Python, SQL, Tableau, Figma), or other skills (e.g. financial modeling, financial statement analysis, accounting")
        interests = st.text_area("Personal Interests/Goals", value="", placeholder="What motivates you? e.g., Solving puzzles, building automation scripts, financial modeling")
        submitted = st.form_submit_button("Get My Personalized Plan")

    if submitted:
        profile_input = f"""
        Name: {student_name}
        Academics/Major: {academics}
        Courses/Skills: {courses}
        Interests/Goals: {interests}
        """

        st.subheader(f"Hello, {student_name}! Agents are analyzing your profile...")

        # --- Task Definitions (Moved inside the execution block to be flexible) ---
        recommendation_task = Task(
            description=dedent(f"""
                Analyze the student profile data below and recommend the single best-fit job role from the knowledge base.
                --- STUDENT PROFILE ---
                {profile_input}
                """),
            expected_output="A single job title that is the best fit for the student (e.g., 'Software Engineer').",
            agent=recommender_agent
        )

        guidance_task = Task(
            description="""
                Use the recommended Job Role from the first task. Use the Career_Knowledge_Retriever and Course Catalog Search tools
                to find specific requirements, courses, and interview tips.
                Format the output into a structured action plan.
            """,
            expected_output="A detailed, markdown-formatted career action plan with sections: Required Skills, Suggested Courses (mentioning availability), and Interview Action Plan.",
            context=[recommendation_task],
            agent=preparer_agent
        )

        career_crew = Crew(
            agents=[recommender_agent, preparer_agent],
            tasks=[recommendation_task, guidance_task],
            process=Process.sequential,
            verbose=2
        )

        with st.spinner("The AI Coach Agents are collaborating to generate your plan..."):
            result = career_crew.kickoff()

            st.session_state["plan_generated"] = True
            st.session_state["full_plan"] = result
            st.session_state["chat_history"] = [{"role": "plan", "content": result}]

            st.success("✅ Analysis Complete!")
            st.markdown("---")
            st.header("2. Your Personalized Career Action Plan")
            st.markdown(result)
            st.markdown("---")
            st.info("Now head over to the 'Chat with Coach' tab for follow-up questions!")

with tab2: # RESUME TAB
    st.header("Resume Review and Revision")
    st.markdown("Upload your resume for AI critique or provide a job link for targeted revision.")

    # File Uploader
    uploaded_file = st.file_uploader("Upload your Resume (PDF or DOCX)", type=["pdf", "docx"], key="resume_uploader")

    # Conditional Form for Critique (Always available)
    with st.container():
        st.subheader("1. AI Resume Critique")
        critique_submitted = st.button("Get Resume Critique", disabled=uploaded_file is None)

        if uploaded_file and critique_submitted:
            # 1. Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                # 2. Define the Critique Task
                critique_task = Task(
                    description=dedent(f"""
                        Read the uploaded resume at '{tmp_file_path}' using the Resume Content Reader tool.
                        Provide a thorough critique in three sections:
                        1. **Overall Impact Score (1-10):** A single score and one sentence justification.
                        2. **Actionable Feedback:** Specific, bulleted advice on sections like 'Experience', 'Skills', and 'Summary'.
                        3. **ATS/Keyword Advice:** Feedback on how well the resume aligns with general professional terms (use the Career_Knowledge_Retriever).
                        The final output must be in clear, detailed Markdown.
                    """),
                    expected_output="A structured, markdown-formatted critique (Score, Actionable Feedback, ATS/Keyword Advice).",
                    agent=critique_agent
                )

                # 3. Run the Critique Crew
                critique_crew = Crew(
                    agents=[critique_agent],
                    tasks=[critique_task],
                    process=Process.sequential,
                    verbose=1
                )

                with st.spinner("The Critique Agent is analyzing your resume..."):
                    critique_result = critique_crew.kickoff()

                st.session_state["critique_result"] = critique_result
                st.success("✅ Resume Critique Complete!")
                st.markdown("---")
                st.markdown("### Your Resume Critique")
                st.markdown(critique_result)

            except Exception as e:
                st.error(f"An error occurred during critique: {e}")
            finally:
                # 4. Clean up the temporary file
                os.unlink(tmp_file_path)

    st.markdown("---")

    # Conditional Form for Revision (Requires file and URL)
    with st.form("resume_revision_form"):
        st.subheader("2. Job-Targeted Resume Revision")
        job_post_url = st.text_input("Paste Job Description URL Here:", key="job_url_input")
        revision_submitted = st.form_submit_button("Generate Targeted Revision", disabled=uploaded_file is None or not job_post_url)

        if uploaded_file and job_post_url and revision_submitted:
            # 1. Save uploaded file temporarily (re-run as it's a new submission)
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                # 2. Define the Revision Task
                # 🎯 FIX: Explicitly tell the agent to use the scraper on the URL input
                revision_task = Task(
                    description=dedent(f"""
                        1. Use the Job Description Scraper tool on the URL provided: '{job_post_url}'.
                        2. Use the Resume Content Reader tool on the resume file: '{tmp_file_path}'.
                        3. Based on the job description text retrieved from the URL and the current resume, revise ONLY the 'Summary/Objective' and ONE 'Experience/Project' bullet point.
                        4. The revision must maximize keyword alignment and professional impact.
                        5. Output should be a side-by-side comparison of the ORIGINAL text and the REVISED text for the two selected sections.
                    """),
                    expected_output="A highly targeted, markdown-formatted document showing the ORIGINAL and REVISED versions of the Summary and one Experience/Project section.",
                    agent=revision_agent
                )

                # 3. Run the Revision Crew
                revision_crew = Crew(
                    agents=[revision_agent],
                    tasks=[revision_task],
                    process=Process.sequential,
                    verbose=1
                )

                with st.spinner("The Revision Agent is generating your targeted updates..."):
                    revision_result = revision_crew.kickoff()

                st.success("✅ Targeted Revision Complete!")
                st.markdown("---")
                st.markdown("### Targeted Resume Revision")
                st.markdown(revision_result)

            except Exception as e:
                st.error(f"An error occurred during revision: {e}")
            finally:
                # 4. Clean up the temporary file
                os.unlink(tmp_file_path)

with tab3: # Existing Chat Tab
    st.header("2. Chat with Your Coach")

    if not st.session_state["plan_generated"] and st.session_state["critique_result"] is None:
        st.info("Please generate your career plan or a resume critique to activate the coach.")

    # Ensure critique is added to chat history once
    if st.session_state["critique_result"] and not any(m.get("role") == "critique" for m in st.session_state["chat_history"]):
         st.session_state["chat_history"].insert(0, {"role": "critique", "content": st.session_state["critique_result"]})

    # Display chat messages from history
    for message in st.session_state["chat_history"]:
        if message["role"] == "plan":
             st.info("The Coach has saved your plan for reference.")
        elif message["role"] == "critique":
             st.info("The Coach has saved your resume critique for reference.")
        elif message["role"] == "assistant":
             st.chat_message("assistant").markdown(message["content"])
        elif message["role"] == "user":
             st.chat_message("user").markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask your coach about your plan, resume critique, or next steps..."):
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- Conversational Task Context ---
        plan_context = st.session_state.get("full_plan", "No career plan generated yet.")
        critique_context = st.session_state.get("critique_result", "No resume critique generated yet.")

        full_context_prompt = f"""
        You are the Interactive Career Coach.
        The student's full career plan (if generated) and resume critique (if generated) are below.
        Use this context and your tools to answer the student's question.

        --- SAVED CAREER PLAN ---
        {plan_context}

        --- SAVED RESUME CRITIQUE ---
        {critique_context}

        --- STUDENT QUESTION ---
        {prompt}

        Answer the student's question concisely based on the context and your tools.
        """

        chat_task = Task(
            description=full_context_prompt,
            expected_output="A concise, supportive, and informative answer to the student's question.",
            agent=chat_agent
        )

        # Run the chat agent
        with st.spinner("Coach is thinking..."):
            chat_crew_fixed.tasks = [chat_task]
            response = chat_crew_fixed.kickoff()

            # Add assistant response to history and display
            st.session_state["chat_history"].append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):

                st.markdown(response)
