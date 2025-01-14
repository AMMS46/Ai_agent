from crewai import Agent
from crewai import Crew, Task, Agent
from crewai_tools import SerperDevTool,SeleniumScrapingTool
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv


from crewai import LLM
os.environ['GEMINI_API_KEY'] = os.getenv("GEMINI_API_KEY")

llm=LLM(
    model=os.getenv("GEMINI_LLM_MODEL"),
    verbose=True,
    google_api_key=os.getenv("GEMINI_API_KEY"),
    max_tokens=5000

)
os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')

#tools
tool = SerperDevTool()
tool_kag = SeleniumScrapingTool()

# Researcher Agent
researcher_agent = Agent(
    llm=llm,
    role="Researcher: Gathers company and industry insights.",
    goal="Collect data about the company's industry segment, key offerings, and strategic focus areas.",
     backstory="""
        This agent is an experienced market researcher specializing in analyzing industries and companies. 
        It has expertise in identifying trends, opportunities, and challenges across various domains to provide actionable insights.
    """,
    tools=[tool, tool_kag],
    memory=True,
    verbose=1
)

# Analyst Agent
analyst_agent = Agent(
    llm=llm,
    role="Analyst: Processes research findings.",
    goal="Analyze research data to identify AI/ML use cases tailored to the company's needs.",
    backstory="""
        This agent is a seasoned data analyst with expertise in AI/ML applications. 
        It excels at identifying actionable use cases by interpreting research findings and aligning them with strategic goals.
    """,
    tools=[tool],
    memory=True,
    verbose=1
)

# Proposal Agent
proposal_agent = Agent(
    llm=llm,
    role="Proposal Writer: Creates a final report.",
    goal="Generate a detailed proposal listing prioritized AI/ML use cases with feasibility analysis.",
     backstory="""
        This agent is a professional technical writer with a strong understanding of AI/ML concepts. 
        It specializes in creating structured reports and proposals that clearly communicate complex ideas to stakeholders.
    """,
    tools=[tool],
    memory=True,
    verbose=1
)


# Task 1: Research Company Information
task1 = Task(
    description="Conduct in-depth research on {company}.",
    agent=researcher_agent,
    expected_output="Detailed summary of industry segment, key offerings, and strategic focus areas.",
    output_file="task1output.txt"
)

# Task 2: Analyze Research Findings
task2 = Task(
    description="Analyze research findings to identify AI/ML use cases.",
    agent=analyst_agent,
    expected_output="List of actionable AI/ML use cases with feasibility analysis.",
    output_file="task2output.txt"
)

# Task 3: Generate Final Proposal
task3 = Task(
    description="Create a final proposal listing top AI/ML use cases.",
    agent=proposal_agent,
    expected_output="Structured report with prioritized use cases and references.Include references to credible sources for each use case.",
    output_file="final_proposal.txt"
)


crew = Crew(
    agents=[researcher_agent, analyst_agent, proposal_agent],
    tasks=[task1, task2, task3],
    verbose=1
)

result = crew.kickoff(inputs={'company': 'AI Planet'})
print(result)
