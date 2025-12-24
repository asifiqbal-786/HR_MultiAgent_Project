import os
import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

## ========= CONFIG & LOGGING ========= ##

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g. https://ai-services-...cognitiveservices.azure.com/
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")        # deployment name for gpt-4o-mini
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

DATA_DIR = DATA_DIR = r"C:\Users\aisiq\OneDrive\Desktop\Recruitment_Data_Analysis"

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

# ========= DATAFRAME BUILD ========= #

print("[LOG] Loading CSV files into dataframes...")

application = pd.read_excel(os.path.join(DATA_DIR, "Application_Table_100.xlsx"))
candidate = pd.read_excel(os.path.join(DATA_DIR, "Candidate_Table_100.xlsx"))
interview = pd.read_excel(os.path.join(DATA_DIR, "Interview_Table_100.xlsx"))
offer = pd.read_excel(os.path.join(DATA_DIR, "Offer_Table_100.xlsx"))
recruiter = pd.read_excel(os.path.join(DATA_DIR, "Recruiter_Table_100.xlsx"))
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


# ========= AGENT PROMPTS ========= #

SUPERVISOR_KNOWLEDGE = """
You orchestrate recruitment analytics queries over a combined dataframe built from:
- application_table_100(application_id, candidate_id, requirement_id, screened_by_recruiter_id,
  current_stage, stage_changed_date, screening_score, status)
- candidate_table_100(candidate_id, candidate_full_name, candidate_email, candidate_phone,
  candidate_skills, candidate_experience_years, candidate_source_of_hire,
  candidate_application_date, candidate_gender, candidate_location)
- interview_table_100(interview_id, application_id, interview_date, interview_round,
  interviewer_id, interview_status, interview_completed_date)
- offer_table_100(offer_id, offer_candidate_id, offer_date, offer_status,
  offer_acceptance_date, Candidate_start_date, Candidate_actual_start_date)
- recruiter_table_100(recruiter_id, recruiter_Name, recruiter_Email,
  recruiter_department, recruiter_status)
- Recruitement_table_100(requirement_id, requirement_job_title, requirement_department,
  requirement_status, requirement_created_date, requirement_target_fill_date,
  requirement_filled_date)

Keys and joins:
- application_table_100.candidate_id -> candidate_table_100.candidate_id
- application_table_100.requirement_id -> Recruitement_table_100.requirement_id
- application_table_100.screened_by_recruiter_id -> recruiter_table_100.recruiter_id
- interview_table_100.application_id -> application_table_100.application_id
- offer_table_100.offer_candidate_id -> candidate_table_100.candidate_id
- offers and interviews are aggregated per application_id.

Classify user questions:
- If they require reading or aggregating these tables/fields, route to SpecialistHRAgent.
- Otherwise, route to GenericHRAgent.

Return JSON with fields:
- "route": "specialist" or "generic"
- "enriched_query": natural language query rewritten with table/field names and filters.
"""

SPECIALIST_SYSTEM = """
You are SpecialistHRAgent.
You understand the recruitment schema and must create SQL ONLY over the raw tables
(application_table_100, candidate_table_100, interview_table_100, offer_table_100,
 recruiter_table_100, Recruitement_table_100).

Return strict JSON:
{
  "sql": "<SQL query>",
  "intent": "<short description>",
  "assumptions": "<clarifications/assumptions>"
}
Do NOT return anything except valid JSON.
"""

GENERIC_SYSTEM = """
You are GenericHRAgent.
If you are called, it means the user's question cannot be answered from the dataframe.
1) Explicitly say that you are answering as generic HR expert, not from data.
2) Give a concise, practical HR/recruitment answer.
"""

FINAL_ANSWER_SYSTEM = """
You are FinalAnswerAgent.
You receive either:
- results from a SQL query (as a small table plus context), OR
- a generic HR explanation.
Your tasks:
1) Present a friendly, concise answer.
2) Mirror the user's technical tone.
3) Always praise the user for the thoughtful question and effort.
4) Propose 3 follow-up questions to deepen the analysis.
Keep the message within a few paragraphs, plus bullet points if helpful.
"""

# ========= SIMPLE EXECUTION HELPERS ========= #

def supervisor_route(user_query: str):
    print("[LOG] Supervisor routing...")
    prompt = f"""
User question: {user_query}

Decide whether this depends on the recruitment tables or is generic HR.
Remember to use the field and table names described above.
"""
    raw = call_llm(SUPERVISOR_KNOWLEDGE, prompt)
    import json
    try:
        data = json.loads(raw)
    except Exception:
        print("[LOG] Failed to parse supervisor JSON, defaulting to specialist.")
        data = {"route": "specialist", "enriched_query": user_query}
    return data["route"], data["enriched_query"]


def specialist_answer(enriched_query: str):
    print("[LOG] SpecialistHRAgent generating SQL...")
    raw = call_llm(SPECIALIST_SYSTEM, enriched_query)

    import json
    spec = json.loads(raw)
    sql = spec["sql"]
    print("[LOG] Generated SQL:\n", sql)

    # Here we run SQL with pandas using duckdb (recommended for quick testing)
    import duckdb
    con = duckdb.connect()
    con.register("application_table_100", application)
    con.register("candidate_table_100", candidate)
    con.register("interview_table_100", interview)
    con.register("offer_table_100", offer)
    con.register("recruiter_table_100", recruiter)
    con.register("Recruitement_table_100", requirement)

    print("[LOG] Executing SQL via duckdb...")
    result_df = con.execute(sql).df()
    print("[LOG] SQL executed. Rows:", len(result_df))

    # Limit preview rows
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


# ========= CONVERSATIONAL AGENT LOOP ========= #

CONVERSATIONAL_SYSTEM = """
You are ConversationalAgent for a recruitment analytics assistant.

Responsibilities:
- Light initial chit-chat.
- Explain that you can answer questions about recruitment pipeline, candidates,
  interviews, offers, recruiters, and requirements.
- If the user asks anything outside HR / recruitment, politely refuse and remind
  them not to misuse this agent.
- If the user asks about HR recruitment or related analytics, you SHOULD NOT
  answer directly. Instead, respond with:
  ROUTE_TO_SUPERVISOR: <cleaned_question>
  and nothing else.
"""

def conversational_turn(user_query: str):
    print("[LOG] ConversationalAgent handling input...")
    content = call_llm(CONVERSATIONAL_SYSTEM, user_query)
    if content.startswith("ROUTE_TO_SUPERVISOR:"):
        cleaned = content.split("ROUTE_TO_SUPERVISOR:", 1)[1].strip()
        return True, cleaned, content
    else:
        # Direct chit-chat / refusal reply
        return False, None, content


def main():
    print("Recruitment Multi-Agent Test CLI")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("You: ").strip()
        if user_query.lower() in {"exit", "quit"}:
            break

        # Step 1: Conversational agent
        route_needed, cleaned_question, conv_reply = conversational_turn(user_query)
        print("\n[ConversationalAgent]\n", conv_reply, "\n")

        if not route_needed:
            continue  # chit-chat / refusal only

        # Step 2: Supervisor
        route, enriched = supervisor_route(cleaned_question)
        print(f"[LOG] Supervisor decided route='{route}'")
        print("[LOG] Enriched query:", enriched)

        # Step 3: Specialist or Generic
        if route == "specialist":
            context = specialist_answer(enriched)
        else:
            context = generic_answer(enriched)

        # Step 4: Final answer
        answer = final_answer(context)
        print("\n[Assistant]\n", answer, "\n")

if __name__ == "__main__":
        main()
