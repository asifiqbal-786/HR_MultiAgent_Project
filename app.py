import os
import pandas as pd
import sqlite3
import re
from openai import AzureOpenAI

from dotenv import load_dotenv
load_dotenv()  # this reads ..env in the current folder

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

DATA_DIR = r"C:\Users\aisiq\OneDrive\Desktop\Recruitment_Data_Analysis"

print("[LOG] Initializing Azure OpenAI client...")
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

def call_llm(system_prompt, user_content):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    resp = client.chat.completions.create(
        model=AZURE_OPENAI_MODEL,
        messages=messages,
        temperature=0
    )
    return resp.choices[0].message.content.strip()
#%%
print("[LOG] Loading Excel files into dataframes...")

# Changed file extensions from .csv to .xlsx
application = pd.read_excel(os.path.join(DATA_DIR, "Application_table_100.xlsx"))
candidate = pd.read_excel(os.path.join(DATA_DIR, "Candidate_table_100.xlsx"))
interview = pd.read_excel(os.path.join(DATA_DIR, "interview_table_100.xlsx"))
offer = pd.read_excel(os.path.join(DATA_DIR, "offer_table_100.xlsx"))
recruiter = pd.read_excel(os.path.join(DATA_DIR, "recruiter_table_100.xlsx"))
requirement = pd.read_excel(os.path.join(DATA_DIR, "Requirement_Table_100.xlsx"))

print("[LOG] Joining base tables...")

df = (
    application
    .merge(candidate, on="candidate_id", how="left")
    .merge(requirement, on="requirement_id", how="left")
    .merge(recruiter, left_on="screened_by_recruiter_id", right_on="recruiter_id", how="left")
)

print("[LOG] Aggregating interviews...")
interview_agg = (
    interview
    .groupby("application_id")
    .agg(
        total_interviews=("interview_id", "count"),
        last_interview_date=("interview_date", "max"),
    )
    .reset_index()
)

print("[LOG] Aggregating offers...")
offers_with_app = offer.merge(
    application[["application_id", "candidate_id"]],
    left_on="offer_candidate_id",
    right_on="candidate_id",
    how="left",
)

offer_agg = (
    offers_with_app
    .groupby("application_id")
    .agg(
        total_offers=("offer_id", "count"),
        last_offer_date=("offer_date", "max"),
    )
    .reset_index()
)

print("[LOG] Final join with interviews and offers...")
df = (
    df
    .merge(interview_agg, on="application_id", how="left")
    .merge(offer_agg, on="application_id", how="left")
)

print("[LOG] Combined dataframe ready. Shape:", df.shape)
df.head()

SUPERVISOR_KNOWLEDGE = """
You are SupervisorAgent for a recruitment analytics assistant.

You must:
1) Decide if the user question REQUIRES the structured recruitment dataframe df
   (tables: applications, candidates, interviews, offers, recruiters, requirements),
   or if it is a generic HR/recruitment best‑practice question.
   - Data question examples: counts, rates, time to fill, breakdowns by department,
     lists of candidates, anything that mentions specific columns like
     current_stage, requirement_department, screening_score, etc.
   - Generic question examples: typical skills for a role, how to design an interview
     process, how to write a JD, how to improve employer branding, etc.

2) Return ONLY valid JSON, no markdown, no explanations:

{
  "route": "specialist" or "generic",
  "enriched_query": "<user query rewritten with any useful clarifications>"
} 

Rules:
- If the user asks about generic qualifications, typical skills, interview design,
  or anything that can be answered without reading df, set "route" to "generic".
- If the user clearly wants numbers or concrete data from the stored tables, set
  "route" to "specialist".
- Do NOT include comments or extra keys. Only the JSON object above.
"""

SPECIALIST_SYSTEM = """
You do NOT write SQL.
You ONLY write Python code that uses an existing pandas DataFrame called df
which already contains all joined recruitment data with columns such as:
- current_stage, stage_changed_date, status, screening_score
- requirement_department, requirement_job_title, requirement_status, ...
- candidate_full_name, candidate_location, candidate_skills, ...
- recruiter_Name, recruiter_department, ...
- total_interviews, last_interview_date, total_offers, last_offer_date, ...

Return ONLY valid JSON:

{
  "code": "<python code that defines a variable named result_df>",
  "intent": "<short description>",
  "assumptions": "<clarifications/assumptions>"
}

Rules:
- The code MUST read from the existing df and end by assigning the final table
  or series to a pandas DataFrame variable called result_df.
- Do not print or plot.
- Do not import modules (assume pandas is already imported as pd).
- No backticks, no ``````, only pure JSON.
"""

GENERIC_SYSTEM = """
You are GenericHRAgent.
If you are called, it means the user's question cannot be answered from the dataframe.
1) Explicitly say that you are answering as generic HR expert, not from data.
2) Give a concise, practical HR/recruitment answer.
"""

FINAL_ANSWER_SYSTEM = """
You are FinalAnswerAgent.
You receive either SQL results (preview table plus context) or a generic HR explanation.
1) Present a friendly, concise answer.
2) Mirror the user's technical tone.
3) Praise the user's thoughtful question and effort.
4) Propose 3 follow-up questions to deepen the analysis.
"""

CONVERSATIONAL_SYSTEM = """
You are ConversationalAgent for a recruitment analytics assistant.

Responsibilities:
- Light initial chit-chat.
- Explain that you can answer questions about recruitment pipeline, candidates,
  interviews, offers, recruiters, and requirements.
- If the user asks anything outside HR / recruitment, politely refuse and remind
  them not to misuse this agent.
- If the user asks about HR recruitment or related analytics, you MUST NOT
  answer directly. Instead, respond with:
  ROUTE_TO_SUPERVISOR: <cleaned_question>
  and nothing else.
"""
import json

def supervisor_route(user_query: str):
    print("[LOG] Supervisor routing...")
    prompt = f"User question: {user_query}\nClassify and enrich as described."
    raw = call_llm(SUPERVISOR_KNOWLEDGE, prompt)
    try:
        data = json.loads(raw)
    except Exception:
        print("[LOG] Failed to parse supervisor JSON, defaulting to specialist.")
        data = {"route": "specialist", "enriched_query": user_query}
    return data["route"], data["enriched_query"]


def specialist_answer(enriched_query: str):
    print("[LOG] SpecialistHRAgent generating pandas code...")
    raw = call_llm(SPECIALIST_SYSTEM, enriched_query)
    print("[DEBUG] Raw specialist response:\n", raw)

    try:
        spec = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            raise ValueError("SpecialistHRAgent did not return JSON:\n" + raw)
        spec = json.loads(match.group(0))

    code = spec["code"]
    print("[LOG] Generated pandas code:\n", code)

    # Prepare an execution namespace with df and pd
    ns = {"df": df.copy(), "pd": pd}

    try:
        exec(code, ns)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Executing generated code failed: {e}\nCode was:\n{code}")

    if "result_df" not in ns:
        raise RuntimeError("Generated code did not create result_df.\nCode was:\n" + code)

    result_df = ns["result_df"]
    print("[LOG] Code executed. Rows in result_df:", len(result_df))

    preview = result_df.head(20).to_markdown(index=False)

    context_for_final = f"""
Intent: {spec.get('intent','')}
Assumptions: {spec.get('assumptions','')}
Preview of results (max 20 rows):

{preview}
"""
    return context_for_final


def generic_answer(enriched_query: str):
    print("[LOG] GenericHRAgent answering...")
    return call_llm(GENERIC_SYSTEM, enriched_query)


def final_answer(context: str):
    print("[LOG] FinalAnswerAgent composing response...")
    return call_llm(FINAL_ANSWER_SYSTEM, context)


def conversational_turn(user_query: str):
    print("[LOG] ConversationalAgent handling input...")
    content = call_llm(CONVERSATIONAL_SYSTEM, user_query)
    if content.startswith("ROUTE_TO_SUPERVISOR:"):
        cleaned = content.split("ROUTE_TO_SUPERVISOR:", 1)[1].strip()
        return True, cleaned, content
    else:
        return False, None, content
    
def ask_recruitment(question: str):
    print("You:", question)

    # Conversational agent
    route_needed, cleaned_question, conv_reply = conversational_turn(question)
    print("\n[ConversationalAgent]\n", conv_reply, "\n")
    if not route_needed:
        return
    # Supervisor
    route, enriched = supervisor_route(cleaned_question)
    print(f"[LOG] Supervisor decided route='{route}'")
    print("[LOG] Enriched query:", enriched)

    # Specialist or generic
    if route == "specialist":
        context = specialist_answer(enriched)
    else:
        context = generic_answer(enriched)

    # Final answer
    answer = final_answer(context)
    print("\n[FinalAnswerAgent]\n", answer)

# -------------------- MAIN LOOP --------------------
if __name__ == "__main__":
    while True:
        user_input = input("Enter your recruitment question (or type 'exit' to quit): ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        ask_recruitment(user_input)

    

    
    
