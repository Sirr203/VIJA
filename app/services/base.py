"""Interface cho từng khâu. Đổi nhà cung cấp = viết thêm class, không sửa chỗ khác."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class SpeechRecognizer(ABC):
    @abstractmethod
    def transcribe(self, audio: bytes, mime: str, language: str) -> str:
        """Trả về chữ đã nhận dạng, chuỗi rỗng nếu không có giọng nói."""


class Translator(ABC):
    tone: str = "auto"

    @abstractmethod
    def translate(
        self, text: str, src: str, dst: str, context: list[str] | None = None, tone: str | None = None
    ) -> str:
        """Dịch text từ src sang dst; context là các câu nói trước đó."""


@dataclass
class Interpretation:
    original: str
    translated: str
    timings_ms: dict[str, int] = field(default_factory=dict)


class Interpreter(ABC):
    """Âm thanh -> (câu gốc, câu dịch). Có thể ghép 2 khâu hoặc gọi 1 lần."""

    tone: str = "auto"

    @abstractmethod
    def interpret(
        self, audio: bytes, mime: str, src: str, dst: str, context: list[str] | None = None, tone: str | None = None
    ) -> Interpretation: ...
