import os
from typing import Literal
from typing_extensions import TypedDict

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="HR Candidate Screening", page_icon="🧑‍💼", layout="centered")
st.title("🧑‍💼 HR Candidate Screening Agent")
st.caption("Built with LangGraph + OpenAI. Paste a job application and get a screening decision.")

# ---------------------------------------------------------------------------
# API key: Streamlit secrets -> env var -> sidebar input
# ---------------------------------------------------------------------------
api_key = None
try:
    api_key = st.secrets.get("OPENAI_API_KEY")
except Exception:
    pass
api_key = api_key or os.environ.get("OPENAI_API_KEY")

with st.sidebar:
    st.header("Settings")
    if not api_key:
        api_key = st.text_input("OpenAI API key", type="password")
    model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
    st.markdown("**Job role:** Python Developer")

if not api_key:
    st.info("Add your OpenAI API key in the sidebar (or in Streamlit secrets) to begin.")
    st.stop()


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------
class State(TypedDict):
    application: str
    experience_level: str
    skill_match: str
    response: str


@st.cache_resource(show_spinner=False)
def build_app(key: str, model: str):
    llm = ChatOpenAI(model=model, temperature=0, api_key=key)

    def categorize_experience(state: State) -> State:
        prompt = ChatPromptTemplate.from_template(
            "Based on the following job application, categorize the candidate as "
            "'Entry-level', 'Mid-level' or 'Senior-level'.\n"
            "Respond with only one of these three labels.\n"
            "Application : {application}"
        )
        level = (prompt | llm).invoke({"application": state["application"]}).content.strip()
        return {"experience_level": level}

    def assess_skillset(state: State) -> State:
        prompt = ChatPromptTemplate.from_template(
            "Based on the job application for a Python Developer, assess the candidate's skillset.\n"
            "Respond with only 'Match' or 'No Match'.\n"
            "Application : {application}"
        )
        match = (prompt | llm).invoke({"application": state["application"]}).content.strip()
        return {"skill_match": match}

    def schedule_hr_interview(state: State) -> State:
        return {"response": "Candidate has been shortlisted for an HR interview."}

    def escalate_to_recruiter(state: State) -> State:
        return {"response": "Candidate has senior-level experience but doesn't match job skills."}

    def reject_application(state: State) -> State:
        return {"response": "Candidate doesn't meet JD and has been rejected."}

    def route_app(
        state: State,
    ) -> Literal["schedule_hr_interview", "escalate_to_recruiter", "reject_application"]:
        skill = state["skill_match"].strip().lower()
        if skill.startswith("match"):  # "No Match" starts with "no", so it is excluded
            return "schedule_hr_interview"
        if "senior" in state["experience_level"].lower():
            return "escalate_to_recruiter"
        return "reject_application"

    wf = StateGraph(State)
    wf.add_node("categorize_experience", categorize_experience)
    wf.add_node("assess_skillset", assess_skillset)
    wf.add_node("schedule_hr_interview", schedule_hr_interview)
    wf.add_node("escalate_to_recruiter", escalate_to_recruiter)
    wf.add_node("reject_application", reject_application)

    wf.add_edge(START, "categorize_experience")
    wf.add_edge("categorize_experience", "assess_skillset")
    wf.add_conditional_edges(
        "assess_skillset",
        route_app,
        {
            "schedule_hr_interview": "schedule_hr_interview",
            "escalate_to_recruiter": "escalate_to_recruiter",
            "reject_application": "reject_application",
        },
    )
    wf.add_edge("schedule_hr_interview", END)
    wf.add_edge("escalate_to_recruiter", END)
    wf.add_edge("reject_application", END)
    return wf.compile()


app = build_app(api_key, model_name)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
example = "I have 10 years of experience in software engineering with expertise in JAVA"
application_text = st.text_area(
    "Candidate application / resume summary",
    value="",
    height=220,
    placeholder=example,
)

if st.button("Screen candidate", type="primary"):
    if not application_text.strip():
        st.warning("Please enter an application first.")
    else:
        try:
            with st.spinner("Screening candidate..."):
                result = app.invoke({"application": application_text})
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

        st.subheader("Result")
        col1, col2 = st.columns(2)
        col1.metric("Experience level", result["experience_level"])
        col2.metric("Skill match", result["skill_match"])

        decision = result["response"]
        if "shortlisted" in decision:
            st.success(decision)
        elif "senior-level" in decision:
            st.warning(decision)
        else:
            st.error(decision)
