import mlflow
import pandas as pd
from agent_demo import agent_workflow, chat_history
from story_agent_Inference import StoryAgentModel
if __name__ == "__main__":
    topic = "The Adventures of a Curious Cat"
    chat_history.clear()
    
    with mlflow.start_run(run_name="Agent Story Writer Run") as run:
        result = agent_workflow(topic)
        print("Final Story:\n", result["story"])
        mlflow.log_param("final_topic", topic)
        mlflow.log_dict(result, artifact_file="agent_workflow_result.json")
        mlflow.pyfunc.log_model(
            artifact_path="story_agent_model",
            python_model=StoryAgentModel(),
            registered_model_name="story-agent",
            code_paths=["."],  # <-- important: include files so mlflow can import at load time
            input_example=pd.DataFrame({"topic": ["Sample topic"]}),
            pip_requirements=[
                "mlflow",
                "google-genai",
                "python-dotenv",
                "pandas",
                "numpy"
            ]
        )
        
        model_uri = f"runs:/{run.info.run_id}/story_agent_model"
        
        mlflow.register_model(
            model_uri=model_uri,
            name="story-agent",
            tags={"model_type": "llm_agent", "workflow": "iterative_story_generation"},
            await_registration_for=120
        )