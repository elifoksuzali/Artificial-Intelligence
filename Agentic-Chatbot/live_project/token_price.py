# calculate.py
from dataclasses import dataclass, field

# Token detayları
@dataclass
class InputTokensDetails:
    cached_tokens: int = 0

@dataclass
class OutputTokensDetails:
    reasoning_tokens: int = 0

@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    input_tokens_details: InputTokensDetails = field(default_factory=lambda: InputTokensDetails())
    output_tokens: int = 0
    output_tokens_details: OutputTokensDetails = field(default_factory=lambda: OutputTokensDetails())
    total_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.requests += other.requests
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.input_tokens_details.cached_tokens += other.input_tokens_details.cached_tokens
        self.output_tokens_details.reasoning_tokens += other.output_tokens_details.reasoning_tokens

# Basit token hesaplama (1 token ~ 4 karakter)
def count_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)
