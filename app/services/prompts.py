"""Centralized system prompts and translation instructions."""
import os

DEFAULT_TONE = "auto"


def tone_instruction(tone: str | None = "auto") -> str:
    """Returns corresponding translation tone/register instruction:
    - auto: preserves speaker's tone, defaults to polite if unclear
    - polite: always polite (Japanese: です/ます)
    - casual: always casual (Japanese: plain form, Vietnamese: informal pronouns)
    - unrecognized value: defaults to auto
    """
    t = (tone or DEFAULT_TONE).strip().lower()
    if t == "polite":
        return (
            "- Tone and register: always polite. "
            "For Japanese: always use polite forms (です/ます). "
            "For Vietnamese: always use polite and respectful forms."
        )
    elif t == "casual":
        return (
            "- Tone and register: always casual. "
            "For Japanese: always use plain/informal forms (体言止め, 辞書形, drop です/ます). "
            "For Vietnamese: always use casual and friendly pronouns and forms."
        )
    else:  # auto or unknown
        return (
            "- Preserve the speaker's meaning, tone and register: casual speech stays casual, "
            "polite speech stays polite. If the register is unclear, default to polite "
            "(for Japanese: です/ます)."
        )


COMMON_RULES = (
    "- Drop false starts, stutters and fillers (e.g. \"th- will\", \"ờ\", \"ừm\") "
    "instead of translating them.\n"
    "- Transliterate personal names, do not translate their meaning."
)


def translate_rules(tone: str = "auto") -> str:
    return f"{tone_instruction(tone)}\n{COMMON_RULES}"


TRANSLATE_RULES = translate_rules("auto")


def translate_system(src: str, dst: str, tone: str = "auto") -> str:
    return (
        f"You are a professional interpreter. Translate from {src} to {dst}.\n"
        f"{translate_rules(tone)}\n- Output only the translation."
    )


def transcribe_prompt(language: str) -> str:
    return (
        f"Transcribe this audio verbatim. The language is {language}. "
        "Do not translate, summarize or correct grammar. Add punctuation. "
        "If there is no speech, output an empty string. Output only the transcript."
    )


def one_step_system(src: str, dst: str, tone: str = "auto") -> str:
    return (
        f"You are a professional interpreter. The audio is spoken in {src}.\n"
        f"1) 'original': transcribe it verbatim in {src} (keep stutters, add punctuation, "
        "no correction). If there is no speech, use an empty string.\n"
        f"2) 'translated': translate it into {dst}, following these rules:\n{translate_rules(tone)}\n"
        "Return only JSON with the keys 'original' and 'translated'."
    )
