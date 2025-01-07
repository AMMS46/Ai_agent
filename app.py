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





#creating agent
researcher= Agent(
    llm=llm,
    role="""
        Senior Researcher: Responsible for conducting in-depth research on a given company or industry. 
        The agent specializes in:
        - Analyzing the industry segment the company operates in (e.g., Automotive, Manufacturing, Finance, Retail, Healthcare).
        - Identifying the company’s key offerings, such as products, services, or solutions.
        - Understanding strategic focus areas like operations optimization, supply chain management, customer experience enhancement, and innovation.
        - Extracting actionable insights from web tools, reports, and public datasets to provide a comprehensive understanding of the company's positioning.
        The output will serve as a foundation for generating AI/ML use cases tailored to improving business operations and customer satisfaction.
    """,
    goal="Use a Web browser tool to analyze the industry and segment the company is working in. Identify the company’s key offerings and strategic focus areas. Provide insights to inform AI/ML use case generation.",
    backstory="This agent is an experienced market analyst specializing in researching industries and companies. It has expertise in identifying trends and opportunities for leveraging AI/ML technologies to improve processes and customer experiences.",
    #allow_delegation=False,
    tools=[tool,tool_kag],
    verbose=1)

#create a task

task1 = Task(
    description="""
        Conduct in-depth research on {company}. Use reliable sources such as the company's website, industry reports, and news articles to analyze the industry segment the company operates in (e.g., Automotive, Manufacturing, Retail, Healthcare). 
        Identify key offerings such as flagship products/services or solutions. Additionally, extract detailed information about the {company}'s strategic focus areas, including operations optimization, supply chain management, customer experience enhancement, and innovation initiatives. 
        Structure the output into three sections: 
        1) Industry segment; 
        2) Key offerings; 
        3) Strategic focus areas.
    """,
    expected_output="""
        A detailed summary of the {company}'s industry and its strategic focus areas in plain text format. The output should include:
        1) Industry segment (e.g., Healthcare → Pharmaceuticals), including sub-segments and competitors;
        2) Key offerings (e.g., flagship products/services) with unique selling points (USPs);
        3) Strategic focus areas (e.g., customer experience improvement, operational efficiency).
    """,
    output_file="task1output.txt",
    tools=[tool,tool_kag],
    memory=True,
    #allow_delegation=True,
    agent= researcher)   

task2 = Task(
    description="""
        Analyze the research findings from 'task1output.txt' to identify current industry trends, challenges, and opportunities for AI/ML adoption. Based on these insights, generate a list of at least 5-10 actionable AI/ML or Generative AI (GenAI) use cases tailored to the {company}'s strategic focus areas (e.g., operations optimization, supply chain management). 
        For each use case:
          a) Provide a brief description of the solution;
          b) Specify the problem it addresses;
          c) Highlight expected benefits (e.g., operational efficiency, cost reduction);
          d) Discuss feasibility considerations (e.g., required data, technology stack);
          e) Include examples of similar use cases in the industry.
        Provide recommendations for prioritizing these use cases based on potential impact and feasibility.
    """,
    expected_output="""
        A structured report containing:
        1) A summary of key industry trends and challenges relevant to the company;
        2) A list of at least 5-10 actionable AI/ML or GenAI use cases tailored to the {company}'s strategic focus areas. Each use case should include:
           a) A brief description of the solution;
           b) The specific problem it addresses;
           c) Expected benefits (e.g., operational efficiency, cost reduction);
           d) Feasibility considerations (e.g., required data, technology stack);
           e) Examples of similar use cases in the industry.
        3) Recommendations for prioritizing these use cases based on their potential impact and feasibility.
    """,
    output_file="task2output.txt",
    tools=[tool, tool_kag],
    #allow_delegation=True,
    agent=researcher
)


final_proposal_task = Task(
    description="""
        Generate a final proposal that lists the top AI/ML use cases relevant to the {company}'s goals and operational needs. 
        Ensure the use cases are aligned with the {company}'s strategic focus areas and include references to credible sources. 
        Each use case should have:
          1) A brief description of the solution.
          2) The specific problem it addresses.
          3) Expected benefits (e.g., operational efficiency, cost reduction).
          4) Feasibility considerations (e.g., required data, technology stack).
          5) Examples of similar use cases in the industry.
        
        Recommend referring to industry-specific reports and insights from sources like McKinsey, Deloitte, Nexocode, or other credible platforms.
    """,
    expected_output="""
        A structured report containing:
        1) A list of top AI/ML use cases relevant to the company's goals.
        2) For each use case:
           a) A brief description of the solution.
           b) The specific problem it addresses.
           c) Expected benefits (e.g., operational efficiency, cost reduction).
           d) Feasibility considerations (e.g., required data, technology stack).
           e) Examples of similar use cases in the industry.
          
        Include references to credible sources for each use case.
    """,
    output_file="final_proposal.txt",
    tools=[tool, tool_kag],
    #allow_delegation=True,
    agent=researcher)
crew= Crew(agents=[researcher], tasks=[task1,task2,final_proposal_task], verbose=1)

result=crew.kickoff(inputs={'company':'Apple'})
print(result)

