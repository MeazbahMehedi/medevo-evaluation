"""Formatting-only companion scorer; the historical scorer is unchanged."""

from src.scoring import boxed_answer, score_answer


SCORING_ID = "boxed-format-unwrapped-em-v1"


def unwrap_formatting(text):
    """Remove only balanced text/textbf wrappers, including nested wrappers."""
    output, index = [], 0
    while index < len(text):
        marker = next((m for m in (r"\textbf{", r"\text{") if text.startswith(m, index)), None)
        if marker:
            start, depth = index + len(marker), 1
            end = start
            while end < len(text) and depth:
                if text[end] in "{}" and (end == 0 or text[end - 1] != "\\"):
                    depth += 1 if text[end] == "{" else -1
                end += 1
            if depth == 0:
                output.append(unwrap_formatting(text[start:end - 1]))
                index = end
                continue
        output.append(text[index])
        index += 1
    return "".join(output)


def score_formatted(raw_response, reference):
    parsed = boxed_answer(raw_response)
    if parsed is None:
        return score_answer(raw_response, reference)
    return score_answer(r"\boxed{" + unwrap_formatting(parsed) + "}", reference)
