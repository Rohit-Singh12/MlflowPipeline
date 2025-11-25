import mlflow.pyfunc
import pandas as pd
from typing import Any
from agent_demo import agent_workflow

class StoryAgentModel(mlflow.pyfunc.PythonModel):
    """
    The model expects a pandas DataFrame with a 'topic' column (one row).
    """

    def load_context(self, context):
        return None

    def predict(self, context, model_input: pd.DataFrame) -> Any:
        if "topic" not in model_input.columns:
            raise ValueError("Input DataFrame must contain a 'topic' column.")
        topic = model_input["topic"].iloc[0]
        result = agent_workflow(str(topic))
        return pd.Series([result.get("story", "")])
