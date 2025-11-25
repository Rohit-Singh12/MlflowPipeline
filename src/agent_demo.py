from google.genai import types
from google.genai.client import Client
import os
from dotenv import load_dotenv
import mlflow
from mlflow.entities import SpanType
from typing import List, Dict, Any
import json
from common import MLFLOW_URI
import pandas as pd

load_dotenv()

# --- MLFLOW SETUP ---
mlflow.set_experiment("Agent Mlflow Experiment")
mlflow.set_active_model(name="gemini-model-lite")

# --- GEMINI CLIENT SETUP ---
api_key = os.getenv("GOOGLE_API_KEY")
os.environ['GOOGLE_GENAI_USE_VERTEXAI']="FALSE"
client = Client(api_key=api_key)

# --- GEMINI CALLS ---
@mlflow.trace(span_type=SpanType.LLM)
def make_gemini_call(query: str) -> str | None:
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=query
    )
    return response.text

@mlflow.trace(span_type=SpanType.LLM)
def make_gemini_chat(instruction: str, messages) -> str | None:
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(system_instruction=instruction),
        contents=messages
    )
    if not response.candidates:
        return None
    return response.candidates[0].content.parts[0].text

# --- AGENT PROMPT DEFINITIONS ---
@mlflow.trace
def writer_agent_instruction():
    '''Registers the prompt for the writer agent in MLflow.'''
    system_prompt = '''You are a helpful, creative story writer. 
        Given a topic, you will write a short story not more than 150 words about that topic.'''
    
    instruction = mlflow.genai.register_prompt(
        name="writer_agent_prompt",
        template=system_prompt,
        commit_message="Prompt for story agent writer"
    )
    return system_prompt

@mlflow.trace
def evaluate_story_instruction() -> str:
    '''Evaluates the story using Gemini model and returns feedback.'''
    evaluation_prompt = '''Evaluate the following story for creativity, coherence, and engagement. 
    Provide score out of 10 for complete story along with brief comments on strengths and areas for improvement under 50 words.
    DONOT provide any other information apart from the score and comments.
    STICK STRICTLY TO THE FORMAT MENTIONED BELOW.
    DONOT add any additional text or explanation.
    DONOT use ```json blocks
        
    Provide your evaluation in the following format (STRICTLY JSON DICTIONARY FORMAT):
    {"score": 3, "comments": "Its not proper"}
    
    '''
    instruction = mlflow.genai.register_prompt(
        name="story_evaluation_prompt",
        template=evaluation_prompt,
        commit_message="Prompt for story evaluation"
    )
    return evaluation_prompt

# --- CHAT HISTORY MANAGEMENT ---
chat_history: List[types.Content] = []
@mlflow.trace(span_type='func', name='create_chat_message', attributes={"module": "tracing"})
def append_to_chat_history(role: str, content: str):
    chat_history.extend([types.Content(
        role=role,
        parts=[types.Part(text=content)]
    )])

# --- MAIN AGENT WORKFLOW ---
is_story_written = False
@mlflow.trace(span_type=SpanType.AGENT)
def agent_workflow(topic: str) -> Dict[str, Any]:
    '''Orchestrates the agent workflow to write and evaluate a story.'''
    global is_story_written
    writer_prompt = writer_agent_instruction()
    evaluation_prompt = evaluate_story_instruction()
    
    # Ensure active run is available for logging feedback
    current_run = mlflow.active_run()
    if not current_run:
        mlflow.start_run(run_name="Agent Story Writer Run")
        current_run = mlflow.active_run()
        
    append_to_chat_history("user", f"Write a short story about the topic: {topic}")
    
    story = ""
    counter = 0
    
    while not is_story_written and counter < 5:
        print(f"Iteration {counter+1} for story writing with chat history:", chat_history)
        
        # 1. Write the story
        story = make_gemini_chat(writer_prompt, chat_history)
        
        # 2. Add story to history for evaluation
        append_to_chat_history("model", story)
        append_to_chat_history("user", "Evaluate the story returned by the writer.")
        
        print("Calling evaluation agent with chat history:", chat_history)
        
        # 3. Evaluate the story
        res = make_gemini_chat(evaluation_prompt, chat_history)
        print("Evaluation Response:", res)
        
        # 4. Parse the response
        parsed_res = None
        try:
            # Strip whitespace and attempt to load JSON
            parsed_res = json.loads(res.strip())
        except json.JSONDecodeError as e:
            mlflow.log_feedback(
                name="Error",
                value = f"Failed to parse evaluation response as JSON: {res}. Error: {str(e)}",
                trace_id=mlflow.active_run().info.run_id)
            break
        except Exception as e:
            # Catch other unexpected exceptions
            mlflow.log_feedback(
                name="Error",
                value = f"Unexpected error during evaluation parsing: {str(e)}",
                trace_id=mlflow.active_run().info.run_id)
            break
            
        # 5. Check evaluation and continue or stop
        if isinstance(parsed_res, dict):
            comments = parsed_res.get("comments", "")
            try:
                score_str = parsed_res.get("score", 0)
                if isinstance(score_str, str):
                    score = int(score_str.strip())
                else:
                    score = int(score_str)
            except ValueError:
                score = 0
                comments += " Score must be an integer only"
            
            
            
            if score >= 6:
                append_to_chat_history("user", "The story is good. No changes needed.")
                is_story_written = True
            else:
                append_to_chat_history("user", f"The story needs improvement. Comments: {comments}. Please rewrite the story.")
        else:
            mlflow.log_feedback(
                name="Error",
                value = "Evaluation response is not in expected dictionary format.",
                trace_id=mlflow.active_run().info.run_id)
            break
            
        counter += 1
            
    return {"message": "Story writing completed.", "story": story, "run_id": current_run.info.run_id}

def reset_chat_history():
    '''Clears the global chat history.'''
    global chat_history
    chat_history.clear()
      

        