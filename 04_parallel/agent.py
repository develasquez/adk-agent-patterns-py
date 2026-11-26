import json
import re
import ssl
import urllib.parse
import urllib.request

import certifi
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent

GEMINI_MODEL = "gemini-3.5-flash-lite"


def search_wikipedia(query: str) -> dict:
    """Search Wikipedia articles and return concise snippets for a query."""
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 3,
        }
    )
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "adk-patterns-py/1.0 (educational example)"}
    )
    try:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(req, timeout=10, context=ssl_context) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = [
            {
                "title": item["title"],
                "snippet": re.sub(r"<[^>]+>", "", item.get("snippet", "")),
            }
            for item in data.get("query", {}).get("search", [])
        ]
        return {"results": results}
    except Exception as exc:
        return {"error": str(exc)}


research_tools = [search_wikipedia]

researcher_agent_1 = LlmAgent(
    name="RenewableEnergyResearcher",
    model=GEMINI_MODEL,
    instruction="""You are an AI Research Assistant specializing in energy.
Research the latest advancements in 'renewable energy sources'.
Use the search_wikipedia tool provided.
Summarize your key findings concisely (1-2 sentences).
Output *only* the summary.""",
    description="Researches renewable energy sources.",
    tools=research_tools,
    output_key="renewable_energy_result",
)

researcher_agent_2 = LlmAgent(
    name="EVResearcher",
    model=GEMINI_MODEL,
    instruction="""You are an AI Research Assistant specializing in transportation.
Research the latest developments in 'electric vehicle technology'.
Use the search_wikipedia tool provided.
Summarize your key findings concisely (1-2 sentences).
Output *only* the summary.""",
    description="Researches electric vehicle technology.",
    tools=research_tools,
    output_key="ev_technology_result",
)

# Researcher 3: Carbon Capture
researcher_agent_3 = LlmAgent(
    name="CarbonCaptureResearcher",
    model=GEMINI_MODEL,
    instruction="""You are an AI Research Assistant specializing in climate solutions.
Research the current state of 'carbon capture methods'.
Use the search_wikipedia tool provided.
Summarize your key findings concisely (1-2 sentences).
Output *only* the summary.""",
    description="Researches carbon capture methods.",
    tools=research_tools,
    output_key="carbon_capture_result",
)

parallel_research_agent = ParallelAgent(
    name="ParallelWebResearchAgent",
    sub_agents=[researcher_agent_1, researcher_agent_2, researcher_agent_3],
    description="Runs multiple research agents in parallel to gather information.",
)

merger_agent = LlmAgent(
    name="SynthesisAgent",
    model=GEMINI_MODEL,
    instruction="""You are an AI Assistant responsible for combining research findings into a structured report.

Your primary task is to synthesize the following research summaries, clearly attributing findings to their source areas. Structure your response using headings for each topic. Ensure the report is coherent and integrates the key points smoothly.

**Crucially: Your entire response MUST be grounded *exclusively* on the information provided in the 'Input Summaries' below. Do NOT add any external knowledge, facts, or details not present in these specific summaries.**

**Input Summaries:**

*   **Renewable Energy:**
    {renewable_energy_result}

*   **Electric Vehicles:**
    {ev_technology_result}

*   **Carbon Capture:**
    {carbon_capture_result}

**Output Format:**

## Summary of Recent Sustainable Technology Advancements

### Renewable Energy Findings
(Based on RenewableEnergyResearcher's findings)
[Synthesize and elaborate *only* on the renewable energy input summary provided above.]

### Electric Vehicle Findings
(Based on EVResearcher's findings)
[Synthesize and elaborate *only* on the EV input summary provided above.]

### Carbon Capture Findings
(Based on CarbonCaptureResearcher's findings)
[Synthesize and elaborate *only* on the carbon capture input summary provided above.]

### Overall Conclusion
[Provide a brief (1-2 sentence) concluding statement that connects *only* the findings presented above.]

Output *only* the structured report following this format. Do not include introductory or concluding phrases outside this structure, and strictly adhere to using only the provided input summary content.""",
    description=(
        "Combines research findings from parallel agents into a structured, "
        "cited report, strictly grounded on provided inputs."
    ),
)

root_agent = SequentialAgent(
    name="ResearchAndSynthesisPipeline",
    sub_agents=[parallel_research_agent, merger_agent],
    description="Coordinates parallel research and synthesizes the results.",
)
