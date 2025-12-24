import os
import json
import pandas as pd
import duckdb
from openai import AzureOpenAI
from dotenv import load_dotenv

# ================== ENV ==================

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

DATA_DIR = r"C:\Users\aisiq\OneDrive\Desktop\Recruitment_Data_Analysis"

print("[LOG] Initializing Azure OpenAI client...")
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

def call_llm(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model=AZURE_OPENAI_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()

# ================== LOAD DATA ==================

print("[LOG] Loading Excel files...")

application = pd.read_excel(os.path.join(DATA_DIR, "Application_Table_100.xlsx"))
candidate = pd.read_excel(os.path.join(DATA_DIR, "Candidate_Table_100.xlsx"))
interview = pd.read_excel(os.path.join(DATA_DIR, "Interview_Table_100.xlsx"))
offer = pd.read_excel(os.path.join(DATA_DIR, "Offer_Table_100.xlsx"))
recruiter = pd.read_excel(os.path.join(DATA_DIR, "Recruiter_Table_100.xlsx"))
requirement = pd.read_excel(os.path.join(DATA_DIR, "Requirement_Table_100.xlsx"))

print("[LOG] Data loaded successfully")

# ================== SYSTEM PROMPTS ==================

SPECIALIST_SYSTEM = """
You are a Recruitment Data Analyst.

You MUST generate SQL that works on the following tables ONLY:

- application_table_100
- candidate_table_100
- interview_table_100
- offer_table_100
- recruiter_table_100
- Recruitement_table_100

Rules:
- Use ONLY these tables and columns
- If the question CANNOT be answered from data, return sql as EMPTY STRING
- Return STRICT JSON ONLY

JSON FORMAT:
{
  "sql": "<sql or empty>",
  "intent": "<short intent>"
}
"""

GENERIC_HR_SYSTEM = """
You are a senior HR expert.

The Excel data was insufficient.
1) Clearly say: "Data not available in Excel"
2) Then answer using general HR/recruitment knowledge
Keep it concise and professional.
"""

FINAL_SYSTEM = """
You are an assistant presenting the final answer.
Be clear, polite, and structured.
You receive either SQL results (preview table plus context) or a generic HR explanation.
1) Present a friendly, concise answer.
2) Mirror the user's technical tone.
3) Praise the user's thoughtful question and effort.
4) Propose 3 follow-up questions to deepen the analysis.
"""

CONVERSATIONAL_SYSTEM = """
You are a recruitment analytics assistant.

RULES:
- EVERY recruitment or HR-related question must be routed.
- NEVER refuse.
- NEVER answer directly.
- Always respond with:

ROUTE_TO_SUPERVISOR: <clean question>
"""

# ================== AGENTS ==================

def conversational_agent(user_query):
    response = call_llm(CONVERSATIONAL_SYSTEM, user_query)
    cleaned = response.split("ROUTE_TO_SUPERVISOR:", 1)[1].strip()
    return cleaned

def specialist_agent(user_query):
    print("[LOG] Specialist generating SQL...")
    raw = call_llm(SPECIALIST_SYSTEM, user_query)

    try:
        spec = json.loads(raw)
        sql = spec.get("sql", "").strip()
    except Exception:
        sql = ""

    if not sql:
        return {"data_found": False, "result": None}

    con = duckdb.connect()
    con.register("application_table_100", application)
    con.register("candidate_table_100", candidate)
    con.register("interview_table_100", interview)
    con.register("offer_table_100", offer)
    con.register("recruiter_table_100", recruiter)
    con.register("Recruitement_table_100", requirement)

    try:
        df = con.execute(sql).df()
    except Exception:
        return {"data_found": False, "result": None}

    if df.empty:
        return {"data_found": False, "result": None}

    return {
        "data_found": True,
        "result": df.head(20).to_markdown(index=False)
    }

def generic_hr_agent(user_query):
    return call_llm(GENERIC_HR_SYSTEM, user_query)

def final_answer_agent(text):
    return call_llm(FINAL_SYSTEM, text)

# ================== MAIN LOOP ==================

def main():
    print("\nRecruitment Multi-Agent Chatbot")
    print("Type 'exit' to quit\n")

    while True:
        user_query = input("You: ").strip()
        if user_query.lower() in {"exit", "quit"}:
            break

        # Step 1: Conversational Agent
        cleaned_question = conversational_agent(user_query)

        # Step 2: Specialist (Excel FIRST)
        excel_result = specialist_agent(cleaned_question)

        if excel_result["data_found"]:
            context = f"""
Answer based strictly on Excel data:

{excel_result['result']}
"""
        else:
            print("[LOG] Excel data not available, using Generic HR")
            hr_answer = generic_hr_agent(user_query)
            context = f"""
Data not available in Excel.

Generic HR perspective:
{hr_answer}
"""

        # Step 3: Final Answer
        final = final_answer_agent(context)
        print("\n[Assistant]\n", final, "\n")

if __name__ == "__main__":
    main()
