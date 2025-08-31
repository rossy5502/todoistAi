import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from dotenv import load_dotenv
import google.generativeai as genai
from todoist_api_python.api import TodoistAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

# Load environment variables
load_dotenv()

# Configuration
class Config:
    TODOIST_API_KEY = os.getenv("TODOIST_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL_NAME = "gemini-2.5-flash"  # Using the latest fast model

# Validate configuration
if not Config.TODOIST_API_KEY or not Config.GEMINI_API_KEY:
    raise ValueError("Missing required API keys in .env file. Please check your configuration.")

# Initialize APIs
try:
    todoist = TodoistAPI(Config.TODOIST_API_KEY)
    # Initialize Gemini
    genai.configure(api_key=Config.GEMINI_API_KEY)
    
    # Initialize LangChain with the specified model
    llm = ChatGoogleGenerativeAI(
        model=Config.MODEL_NAME,
        google_api_key=Config.GEMINI_API_KEY,
        temperature=0.3
    )
    
    # Print model info for debugging
    print(f"Using model: {Config.MODEL_NAME}")
    
    # Test the connection
    try:
        llm.invoke("Test connection")
    except Exception as e:
        print("⚠️  Warning: Could not connect to Gemini API. Some features may be limited.")
        print(f"   Error details: {str(e)}")
except Exception as e:
    raise RuntimeError(f"Failed to initialize APIs: {str(e)}")

# Tools for the agent
@tool
def add_task(task_description: str, due_date: Optional[str] = None, priority: int = 1) -> str:
    """Add a new task to Todoist.
    
    Args:
        task_description: The description of the task
        due_date: Optional due date in DD-MM-YYYY format or 'today'/'tomorrow'
        priority: Task priority from 1 (normal) to 4 (urgent)
    """
    try:
        # Format due date if provided
        due_string = None
        if due_date:
            if due_date.lower() == 'today':
                due_string = 'today'
            elif due_date.lower() == 'tomorrow':
                due_string = 'tomorrow'
            else:
                # Try to parse the date if it's in DD-MM-YYYY format
                try:
                    datetime.strptime(due_date, '%d-%m-%Y')
                    due_string = due_date
                except ValueError:
                    pass
        
        # Create task with proper parameters
        task = todoist.add_task(
            content=task_description,
            due_string=due_string,
            priority=priority
        )
        return f"✅ Task added: {task.content} (Due: {task.due.date if task.due else 'No due date'})"
    except Exception as e:
        return f"❌ Error adding task: {str(e)}"

@tool
def list_tasks() -> str:
    """List all active tasks from Todoist."""
    try:
        tasks = todoist.get_tasks()
        if not tasks:
            return "No tasks found in your Todoist account."
            
        task_list = []
        for i, task in enumerate(tasks, 1):
            due_date = f" (Due: {task.due.date}" + (f" {task.due.datetime.split('T')[1][:5]}" if task.due.datetime else '') + ")" if task.due else ""
            task_list.append(f"{i}. {task.content}{due_date}")
            
        return "\n".join(["📋 Your tasks:", *task_list])
    except Exception as e:
        return f"❌ Error fetching tasks: {str(e)}"

@tool
def complete_task(task_id: int) -> str:
    """Mark a task as complete.
    
    Args:
        task_id: The ID of the task to complete (number from the task list)
    """
    try:
        tasks = todoist.get_tasks()
        if 1 <= task_id <= len(tasks):
            task = tasks[task_id - 1]
            todoist.close_task(task_id=task.id)
            return f"✅ Task completed: {task.content}"
        return f"❌ Invalid task ID. Please use a number between 1 and {len(tasks)}"
    except Exception as e:
        return f"❌ Error completing task: {str(e)}"

# Initialize tools
tools = [add_task, list_tasks, complete_task]

# System prompt
system_prompt = """You are a helpful AI assistant that helps manage tasks in Todoist and answers general questions.
You have access to the following tools for task management:
- add_task: Add a new task with optional due date and priority
- list_tasks: List all active tasks
- complete_task: Mark a task as complete

For task-related requests, use the appropriate tool. For general questions, use your knowledge to provide helpful responses.
Be concise and to the point in your responses."""

# Create the agent
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_openai_tools_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def main():
    print("\n🤖 Welcome to your AI Task Manager! Type 'exit' to quit.")
    print("You can ask me to add tasks, list tasks, or ask general questions.")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Goodbye!")
                break
                
            if not user_input:
                continue
                
            # Process the input
            result = agent_executor.invoke({"input": user_input})
            print(f"\n🤖 {result['output']}")
            
        except KeyboardInterrupt:
            print("\n👋 Operation cancelled by user.")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()