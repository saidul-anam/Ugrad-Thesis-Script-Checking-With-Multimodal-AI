import os
import glob
from typing import Optional, Dict, List


class RAGContextProvider:
    """
    Provides domain-specific textbook, syllabus, and thematic reference context
    for Bangla CQ and other subject-specific evaluation tasks.
    """

    def __init__(self, context_dir: str = "configs/context/"):
        self.context_dir = context_dir
        self._contexts: Dict[str, str] = {}
        self._load_contexts()

    def _load_contexts(self):
        if not os.path.exists(self.context_dir):
            return
        for filepath in glob.glob(os.path.join(self.context_dir, "*.txt")):
            topic_name = os.path.splitext(os.path.basename(filepath))[0].lower()
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    self._contexts[topic_name] = f.read().strip()
            except Exception as e:
                print(f"[RAGContextProvider] Warning: Failed to load {filepath}: {e}")

    def get_context(self, topic_or_subject: str) -> Optional[str]:
        """Retrieve preloaded context matching topic or subject."""
        key = topic_or_subject.lower()
        if key in self._contexts:
            return self._contexts[key]
        for k, v in self._contexts.items():
            if k in key or key in k:
                return v
        return None
