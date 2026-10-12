"""Provider-independent LLM wrapper that injects training elite references."""

from __future__ import annotations

import random


class EliteReferenceLLM:
    def __init__(
        self,
        client,
        elites,
        target_size,
        seed=0,
        reference_count=2,
    ):
        if reference_count < 0:
            raise ValueError("reference_count cannot be negative")
        self.client = client
        self.target_size = int(target_size)
        self.batch_size = client.batch_size
        self.model = client.model
        self.temperature = client.temperature
        ordered = sorted(
            elites, key=lambda elite: (elite.source_size, elite.code_sha256)
        )
        count = min(reference_count, len(ordered))
        self.references = tuple(random.Random(seed).sample(ordered, count))
        self.reference_source_sizes = tuple(
            elite.source_size for elite in self.references
        )
        self.reference_code_hashes = tuple(
            elite.code_sha256 for elite in self.references
        )

    def _message(self, message: str) -> str:
        if not self.references:
            return message
        sections = [
            (
                f"Training elite references for target TSP size {self.target_size}. "
                "Use these only as idea/code context; score candidates on the target data."
            )
        ]
        for index, elite in enumerate(self.references, start=1):
            sections.append(
                f"Reference {index} (source size {elite.source_size}, "
                f"code sha256 {elite.code_sha256}):\n"
                f"Idea: {elite.idea}\n"
                f"```python\n{elite.code}\n```"
            )
        sections.append(f"Target request:\n{message}")
        return "\n\n".join(sections)

    def prompt(self, expertise, message, temperature=None):
        return self.client.prompt(expertise, self._message(message), temperature)

    def prompt_batch(self, expertise, message_batch, temperature=None):
        return self.client.prompt_batch(
            expertise,
            [self._message(message) for message in message_batch],
            temperature,
        )
