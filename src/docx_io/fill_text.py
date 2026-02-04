from docx.text.paragraph import Paragraph
from src.docx_io.traverse import TextContainer


def replace_span_across_runs(
    paragraph: Paragraph,
    span_start: int,
    span_end: int,
    replacement: str,
) -> bool:
    runs = paragraph.runs
    if not runs or span_start >= span_end:
        return False

    full_text = "".join(run.text for run in runs)
    if span_start < 0 or span_end > len(full_text):
        return False

    run_spans = []
    current = 0
    for run in runs:
        run_text = run.text
        run_end = current + len(run_text)
        run_spans.append((current, run_end))
        current = run_end

    start_idx = None
    end_idx = None
    for idx, (s, e) in enumerate(run_spans):
        if start_idx is None and s <= span_start < e:
            start_idx = idx
        if s < span_end <= e:
            end_idx = idx
            break

    if start_idx is None or end_idx is None:
        return False

    segments = replacement.split("\n")
    for idx in range(start_idx, end_idx + 1):
        run = runs[idx]
        run_text = run.text
        s, e = run_spans[idx]

        prefix = ""
        suffix = ""
        if idx == start_idx:
            prefix = run_text[: max(0, span_start - s)]
        if idx == end_idx:
            suffix = run_text[max(0, span_end - s) :]

        if idx == start_idx:
            run.text = prefix + (segments[0] if segments else "")
            for segment in segments[1:]:
                run.add_break()
                run.add_text(segment)
            if start_idx == end_idx and suffix:
                run.add_text(suffix)
        elif idx == end_idx:
            run.text = suffix
        else:
            run.text = ""

    return True


def fill_text_container(container: TextContainer, span_start: int, span_end: int, value: str) -> bool:
    paragraph = container.obj
    if not isinstance(paragraph, Paragraph):
        return False
    return replace_span_across_runs(paragraph, span_start, span_end, value)
