"""Push models to Hugging Face Hub with auto-generated model cards."""

from __future__ import annotations

from pathlib import Path


def push_to_hub(
    model_path: str | Path,
    repo_id: str,
    *,
    private: bool = False,
    commit_message: str = "Upload smithery fine-tuned model",
) -> str:
    """Upload a model directory to Hugging Face Hub.

    Generates a model card with training metadata and pushes all files.

    Args:
        model_path: Local path to the model.
        repo_id: Hub repository id (e.g. "user/my-tool-agent").
        private: Whether the Hub repo should be private.
        commit_message: Commit message for the upload.

    Returns:
        URL of the uploaded model on the Hub.
    """
    raise NotImplementedError
