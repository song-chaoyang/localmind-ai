"""
Model manager for OpenMind.

Provides utilities for listing, pulling, and inspecting Ollama models
through the Ollama REST API.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"


class ModelManager:
    """Interface for managing Ollama models.

    All operations go through the Ollama REST API running on the local
    machine.

    Args:
        base_url: Base URL of the Ollama API.

    Example::

        mm = ModelManager()
        models = mm.list_available_models()
        mm.pull_model("mistral")
        info = mm.get_model_info("mistral")
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=httpx.Timeout(30.0, connect=5.0),
        )

    def _check_connection(self) -> None:
        """Verify that Ollama is reachable.

        Raises:
            ConnectionError: If Ollama is not running.
        """
        try:
            resp = self._client.get("/")
            if resp.status_code == 200:
                return
        except (httpx.ConnectError, httpx.ConnectTimeout):
            pass
        raise ConnectionError(
            "Cannot connect to Ollama. Ensure it is running at "
            f"{self._base_url}. Install from https://ollama.ai "
            "and start with 'ollama serve'."
        )

    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all models available locally in Ollama.

        Returns:
            A list of model dictionaries, each containing at least
            ``name``, ``model``, ``modified_at``, and ``size`` keys.

        Raises:
            ConnectionError: If Ollama is unreachable.
        """
        self._check_connection()
        resp = self._client.get("/api/tags")
        resp.raise_for_status()
        data = resp.json()
        return data.get("models", [])

    def pull_model(self, name: str, timeout: float = 600.0) -> Dict[str, Any]:
        """Pull (download) a model from the Ollama model registry.

        This may take a significant amount of time for large models.

        Args:
            name: Name of the model to pull (e.g. ``"llama3"``).
            timeout: HTTP request timeout in seconds.

        Returns:
            The Ollama API response as a dictionary.

        Raises:
            ConnectionError: If Ollama is unreachable.
            httpx.HTTPStatusError: If the pull request fails.
        """
        self._check_connection()
        logger.info("Pulling model '%s' ...", name)
        resp = self._client.post(
            "/api/pull",
            json={"name": name, "stream": False},
            timeout=httpx.Timeout(timeout, connect=10.0),
        )
        resp.raise_for_status()
        logger.info("Model '%s' pulled successfully.", name)
        return resp.json()

    def get_model_info(self, name: str) -> Dict[str, Any]:
        """Retrieve detailed information about a specific model.

        Args:
            name: Name of the model.

        Returns:
            A dictionary with model details such as license, parameters,
            template, and system prompt.

        Raises:
            ConnectionError: If Ollama is unreachable.
            httpx.HTTPStatusError: If the model is not found.
        """
        self._check_connection()
        resp = self._client.post("/api/show", json={"name": name})
        resp.raise_for_status()
        return resp.json()

    def delete_model(self, name: str) -> Dict[str, Any]:
        """Delete a locally stored model.

        Args:
            name: Name of the model to delete.

        Returns:
            The Ollama API response.

        Raises:
            ConnectionError: If Ollama is unreachable.
            httpx.HTTPStatusError: If the deletion fails.
        """
        self._check_connection()
        resp = self._client.delete("/api/delete", json={"name": name})
        resp.raise_for_status()
        logger.info("Model '%s' deleted.", name)
        return resp.json()

    def is_model_available(self, name: str) -> bool:
        """Check whether a model is available locally.

        Args:
            name: Model name to check.

        Returns:
            ``True`` if the model exists locally.
        """
        try:
            models = self.list_available_models()
            model_names = [m.get("name", "") for m in models]
            return name in model_names or f"library/{name}" in model_names
        except ConnectionError:
            return False

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __repr__(self) -> str:
        return f"ModelManager(base_url={self._base_url!r})"

    def __enter__(self) -> "ModelManager":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
