# web_form_agent.py (Updated for Multi-Page Forms)

import os
import asyncio
import traceback  # For detailed error printing
from langchain_experimental.tools.playwright.toolkit import PlayWrightBrowserToolkit

from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from playwright.async_api import async_playwright
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# Import the RAG tool function from main_rag.py
try:
    from main_rag import get_info_from_rag
except ImportError:
    print("Error: Could not import 'get_info_from_rag' from main_rag.py.")
    print("Ensure main_rag.py is in the same directory or Python path.")
    exit()

# --- Configuration ---
# Target URL of the web form
# !!! IMPORTANT: Replace with the actual URL of the form !!!
TARGET_URL = "https://example.com/your-multi-page-form"  # <--- CHANGE THIS

# LLM for the Agent
AGENT_LLM_MODEL = "llama3:8b"  # Or another capable model available via Ollama

# --- Main Agent Logic ---


async def main():
    """Runs the web form filling agent."""
    print("--- Web Form Filling Agent (Multi-Page Support) ---")

    # --- Initialize Agent Components ---
    print(f"Initializing Agent LLM: {AGENT_LLM_MODEL}")
    # Lower temperature for more predictable actions
    llm = ChatOllama(model=AGENT_LLM_MODEL, temperature=0.1)

    # --- Set up Tools ---
    # 1. RAG Tool
    print("Setting up RAG tool...")
    rag_tool = Tool(
        name="GetPersonalInfo",
        func=get_info_from_rag,
        description="Useful for answering questions about personal details, work history, skills, or other information stored in the user's knowledge base (resumes, documents). Input should be a question asking for a specific piece of information (e.g., 'What is my email address?', 'What was my job title at Company Y?').",
    )
    print("RAG tool created.")

    # 2. Browser Tools (Async)
    print("Setting up Playwright browser tools...")
    # Initialize Playwright directly with proper async context
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(timeout=30000)

        # Initialize toolkit with browser instance
        browser_toolkit = PlayWrightBrowserToolkit.from_browser(
            async_browser=browser)
        browser_tools = browser_toolkit.get_tools()
        print(
            f"Browser tools created: {[tool.name for tool in browser_tools]}")

        # Combine tools
        all_tools = browser_tools + [rag_tool]

        # --- Create Agent Prompt (UPDATED INSTRUCTIONS) ---
        # Uses ReAct style prompt template
        prompt_template = """
        You are an assistant designed to fill out web forms, potentially spanning multiple pages, based on information from a user's knowledge base.

        You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do. This includes planning steps like navigating, finding fields, getting info, filling fields, finding navigation buttons, clicking, and submitting.
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action (e.g., URL for navigate, CSS selector for find/click/input, question for GetPersonalInfo)
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I have completed all necessary steps and the form is submitted or I have filled all possible fields.
        Final Answer: the final response or a confirmation of completion (e.g., "Form submitted successfully.", "Filled all fields on page 1 and clicked Next.", "Completed filling the form.")

        Your Goal: Fill out the web form located at the specified URL, navigating through pages if necessary.

        Instructions:
        1. Navigate to the target URL: {target_url}
        2. **On the current page:** Carefully examine the page content (using 'current_page' and 'extract_text' if needed) to identify the form fields (e.g., input boxes, text areas, dropdowns). Look for labels near the fields or attributes like 'name', 'id', or 'placeholder'.
        3. For each field identified on the *current page*, determine the required information (e.g., "First Name", "Email Address").
        4. Use the 'GetPersonalInfo' tool to retrieve the necessary information from the knowledge base for each field. Formulate clear questions (e.g., "What is my first name?").
        5. Use the browser tools (like 'find_element', 'input_text') to locate the HTML element for each form field on the *current page* and input the retrieved information. Use specific CSS selectors or XPath. For dropdowns (<select>), try using 'input_text' first; if that fails, find the dropdown and then the specific option to click.
        6. **After filling fields on the current page:** Look for a button to proceed to the next page (common labels: "Next", "Continue", "Proceed"). Use 'find_element' and then 'click_element' on this button.
        7. **If you clicked a 'Next'/'Continue' button:** Repeat steps 2-6 for the new page that loads.
        8. **If you cannot find a 'Next'/'Continue' button:** Look for a final submission button (common labels: "Submit", "Finish", "Complete", "Register"). If found, use 'find_element' and 'click_element' to submit the form.
        9. If you cannot find *any* relevant fields, 'Next' button, or 'Submit' button on a page after navigating, state that you cannot proceed further.
        10. If you cannot find a field or retrieve information for it, make a note of it and continue with the other fields/steps.
        11. Provide a final confirmation when the form is submitted or when you have finished filling all identifiable fields across all pages and cannot find a submit button.

        Begin!

        Target URL: {target_url}
        Input: Start filling the form at the target URL, navigating pages as needed, and submit when complete.
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_template),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # --- Create Agent ---
        print("Creating ReAct agent...")
        agent = create_react_agent(llm, all_tools, prompt)

        # --- Create Agent Executor ---
        print("Creating Agent Executor...")
        agent_executor = AgentExecutor(
            agent=agent,
            tools=all_tools,
            verbose=True,  # Keep True for debugging multi-page logic
            handle_parsing_errors=True,
            max_iterations=25  # Increased max_iterations slightly for multi-page forms
        )

        # --- Run the Agent ---
        print(
            f"\n--- Running Agent to Fill Multi-Page Form at: {TARGET_URL} ---")
        try:
            # Provide the initial input and target URL context
            result = await agent_executor.ainvoke({
                "input": "Navigate to the form, identify and fill fields on the first page, find and click the 'Next' button if it exists, repeat for subsequent pages, and finally find and click the 'Submit' button.",
                "target_url": TARGET_URL,
                # Pass tool names and descriptions dynamically to the prompt
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in all_tools]),
                "tool_names": ", ".join([tool.name for tool in all_tools]),

            })
            print("\n--- Agent Run Finished ---")
            print("Final Result:", result.get(
                'output', 'No output field found.'))

        except Exception as e:
            print(f"\n--- Agent Run Failed ---")
            print(f"Error: {e}")
            traceback.print_exc()
        finally:
            # Close the browser when done
            await browser.close()
            print("Browser closed.")


if __name__ == "__main__":
    # Run the async main function
    # Use appropriate method depending on environment (script vs notebook)
    asyncio.run(main())
