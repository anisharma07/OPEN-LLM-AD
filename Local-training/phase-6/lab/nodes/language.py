"""Question builder, text prompt builder, MLLM, answer parser and the subtask-selective gate."""

import random
import re

from ..data import SUBTASKS, domain_knowledge_for
from ..mllm import MODELS, list_local_models
from ..registry import Node, Param, Port, Result, register

# ----------------------------------------------------------------------------- questions
_LEXICAL = [
    (r"^Is there any defect in the object\?$", "Does this object show any defect?"),
    (r"What is the type of the defect\?", "Which kind of defect is it?"),
    (r"Where is the defect\?", "In which part of the image is the defect located?"),
    (r"What is the appearance of the defect\?", "How does the defect look?"),
    (r"What is the effect of the defect\?", "What consequence does this defect have?"),
    (r"There is a defect in the object\.", "The object has a defect."),
    (r"\bin the object\b", "on this item"),
    (r"\bthe object\b", "the item"),
]

PARAPHRASES = {
    "original": "MMAD wording, unchanged",
    "P1 lexical": "rule-based rewording of MMAD's templated questions",
    "P2 role framing": "prefix: 'You are an industrial quality inspector...'",
    "P3 exam format": "'Question: ... Select the single best option.'",
}


def paraphrase(text, which):
    if which == "P1 lexical":
        out = text
        for pat, rep in _LEXICAL:
            out = re.sub(pat, rep, out)
        return out if out != text else "Look at the image carefully. " + text
    if which == "P2 role framing":
        return "You are an industrial quality inspector examining a product image. " + text
    if which == "P3 exam format":
        return f"Question: {text}\nSelect the single best option."
    return text


@register
class QuestionBuilder(Node):
    TYPE = "QuestionBuilder"
    TITLE = "Question Builder"
    CATEGORY = "Question"
    DESCRIPTION = """
Surface-form perturbations of the MMAD question: 3 paraphrase templates and
option shuffling (the answer key is remapped). Sweep 'paraphrase' to measure
the prompt-sensitivity spread; enable shuffling to expose letter-position bias."""
    INPUTS = [Port("question", "QUESTION")]
    OUTPUTS = [Port("question", "QUESTION")]
    PARAMS = [
        Param("paraphrase", "choice", "original", list(PARAPHRASES)),
        Param("shuffle_options", "bool", False),
        Param("shuffle_seed", "int", 0, min=0, max=1_000_000),
    ]

    def run(self, ctx, inputs, p):
        q = dict(inputs["question"])
        q["question"] = paraphrase(q["question"], p["paraphrase"])
        if p["shuffle_options"] and len(q["options"]) > 1:
            letters = sorted(q["options"])
            texts = [q["options"][L] for L in letters]
            gt_text = q["options"].get(q["answer"])
            rng = random.Random(f"{q['qid']}|{p['shuffle_seed']}")
            rng.shuffle(texts)
            q["options"] = dict(zip(letters, texts))
            if gt_text is not None:
                q["answer"] = letters[texts.index(gt_text)]
        opts = "\n".join(f"({k}) {v}" for k, v in sorted(q["options"].items()))
        return Result(outputs={"question": q}, ui={"text": f"{q['question']}\n{opts}", "metrics": {"GT": q["answer"]}},
                      records=[{"kind": "context", "paraphrase": p["paraphrase"], "shuffled": p["shuffle_options"]}])


# ----------------------------------------------------------------------------- prompts
LETTER_ONLY = "Answer with the letter of the correct option (e.g., A, B, C, or D). Give only the letter."
REASONING = "Briefly describe what you see in the relevant region, then finish with a final line 'Answer: <letter>'."

TEMPLATES = {
    "MMAD vanilla (Arm A)": "{question}\n\n{options}\n\n{answer_instruction}",
    "grounded (Phase 5)": (
        "A vision-expert anomaly detection model has localized the potential flaw area inside the {color} bounding box.\n"
        "Focus your visual inspection on the highlighted {color} region:\n\n{question}\n\n{options}\n\n{answer_instruction}"),
    "grounded + score": (
        "An anomaly detector ({detector}) flagged the region inside the {color} box. Its anomaly score is "
        "{score_norm} times the highest score it produced on defect-free references.\n\n"
        "{question}\n\n{options}\n\n{answer_instruction}"),
    "grounded + location words": (
        "An anomaly detector flagged a candidate defect in the {region} of the image, marked with a {color} box.\n\n"
        "{question}\n\n{options}\n\n{answer_instruction}"),
    "hedged hint": (
        "Hint: an automatic detector marked a {color} box where something may be abnormal. The detector is often "
        "wrong, so verify it against the whole object before answering.\n\n{question}\n\n{options}\n\n{answer_instruction}"),
    "crop note": (
        "The image is a zoomed-in view of a region flagged by an anomaly detector on a {category}.\n\n"
        "{question}\n\n{options}\n\n{answer_instruction}"),
    "custom": "",
}


class _SafeDict(dict):
    def __missing__(self, k):
        return "{" + k + "}"


@register
class TextPrompt(Node):
    TYPE = "TextPrompt"
    TITLE = "Prompt Builder"
    CATEGORY = "Prompt"
    DESCRIPTION = """
Builds the text sent to the MLLM. Placeholders for 'custom': {question}
{options} {answer_instruction} {color} {detector} {score_norm} {region}
{category} {domain_knowledge}. With cue_aware on, the vanilla template is
used whenever the detector did not fire (Phase 5 behaviour)."""
    INPUTS = [Port("question", "QUESTION"), Port("region", "REGION", optional=True)]
    OUTPUTS = [Port("prompt", "TEXT")]
    PARAMS = [
        Param("template", "choice", "grounded (Phase 5)", list(TEMPLATES)),
        Param("custom_template", "textarea", "{question}\n\n{options}\n\n{answer_instruction}", sweepable=False),
        Param("answer_format", "choice", "letter only", ["letter only", "reason then answer"]),
        Param("cue_aware", "bool", True),
        Param("domain_knowledge", "bool", False, help="prepend MMAD's domain_knowledge.json entry for the category"),
        Param("color_word", "choice", "RED", ["RED", "red", "GREEN", "YELLOW", "highlighted"]),
    ]

    def run(self, ctx, inputs, p):
        q, r = inputs["question"], inputs.get("region")
        tpl_name = p["template"]
        fired = bool(r and r.get("fired"))
        if tpl_name != "MMAD vanilla (Arm A)" and p["cue_aware"] and not fired:
            tpl_name = "MMAD vanilla (Arm A)"
        tpl = p["custom_template"] if tpl_name == "custom" else TEMPLATES[tpl_name]
        dk = domain_knowledge_for(q.get("dataset", ""), q.get("category", "")) if p["domain_knowledge"] else ""
        fields = _SafeDict(
            question=q["question"],
            options="\n".join(f"({k}) {v}" for k, v in sorted(q["options"].items())),
            answer_instruction=LETTER_ONLY if p["answer_format"] == "letter only" else REASONING,
            color=p["color_word"], detector=(r or {}).get("detector", "detector"),
            score_norm=f"{r['score_norm']:.2f}" if r and r.get("score_norm") is not None else "n/a",
            region=", ".join((r or {}).get("region_words", [])) or "image",
            category=q.get("category", "object").replace("_", " "), domain_knowledge=dk,
        )
        text = tpl.format_map(fields)
        if dk and "{domain_knowledge}" not in tpl:
            text = f"Domain knowledge about this product:\n{dk.strip()}\n\n{text}"
        return Result(outputs={"prompt": text}, ui={"text": text, "metrics": {"template": tpl_name}},
                      records=[{"kind": "context", "prompt_template": tpl_name}])


# ----------------------------------------------------------------------------- MLLM
@register
class MLLM(Node):
    TYPE = "MLLM"
    TITLE = "Multimodal LLM"
    CATEGORY = "Reason"
    DESCRIPTION = """
Runs a local vision-language model (models found in your Hugging Face cache).
'letter-logits' reads the probability of each option letter from the first
answer token (one forward pass, gives calibrated confidences); 'generate'
decodes text. Only one model is kept in memory; switching models reloads."""
    INPUTS = [Port("image", "IMAGE"), Port("prompt", "TEXT"), Port("question", "QUESTION", optional=True),
              Port("extra_image", "IMAGE", optional=True)]
    OUTPUTS = [Port("response", "RESPONSE")]
    PARAMS = [
        Param("model", "choice", "Qwen/Qwen3-VL-2B-Instruct", list_local_models),
        Param("precision", "choice", "auto", ["auto", "fp16", "bf16", "fp32", "4bit-nf4", "8bit"]),
        Param("device", "choice", "auto", ["auto", "cuda", "cpu"]),
        Param("answer_mode", "choice", "generate", ["generate", "letter-logits"]),
        Param("max_new_tokens", "int", 8, min=1, max=512, step=1),
        Param("temperature", "float", 0.0, min=0.0, max=2.0, step=0.05, help="0 = greedy"),
        Param("top_p", "float", 1.0, min=0.05, max=1.0, step=0.05),
        Param("seed", "int", 0, min=0, max=1_000_000),
        Param("max_pixels", "int", 401408, min=50176, max=2_000_000, step=50176,
              help="image is downscaled to at most this many pixels (401408 = 512 x 28 x 28, Phase 5 setting)"),
        Param("extra_image_first", "bool", True, help="put extra_image (e.g. a normal reference) before the query"),
        Param("system_prompt", "textarea", "", sweepable=False),
    ]

    def run(self, ctx, inputs, p):
        q = inputs.get("question")
        letters = tuple(sorted(q["options"])) if q else ("A", "B", "C", "D")
        imgs = [inputs["image"]]
        if inputs.get("extra_image") is not None:
            imgs = [inputs["extra_image"], inputs["image"]] if p["extra_image_first"] else [inputs["image"], inputs["extra_image"]]
        out = MODELS.answer(
            p["model"], imgs, inputs["prompt"], system=p["system_prompt"], precision=p["precision"],
            device=p["device"], answer_mode=p["answer_mode"], max_new_tokens=p["max_new_tokens"],
            temperature=p["temperature"], top_p=p["top_p"], seed=p["seed"], max_pixels=p["max_pixels"],
            letters=letters, rng_key=(q or {}).get("qid", ""))
        out["model"] = p["model"]
        ui = {"text": out["text"], "metrics": {"latency_s": round(out["latency"], 3), "tokens_in": out["input_tokens"]}}
        if out.get("probs"):
            ui["probs"] = out["probs"]
        return Result(outputs={"response": out}, ui=ui,
                      records=[{"kind": "context", "model": p["model"], "mllm_latency": round(out["latency"], 4)}])


# ----------------------------------------------------------------------------- parsing
def parse_lenient(text, letters):
    """Phase-4 heuristics plus option-text matching."""
    if not text:
        return None
    up = text.strip().upper()
    L = "".join(letters)
    m = re.match(rf"^\(?([{L}])\b", up)
    if m:
        return m.group(1)
    m = re.search(rf"\(([{L}])\)|\b([{L}])[).:]", up)
    if m:
        return m.group(1) or m.group(2)
    found = re.findall(rf"\b([{L}])\b", up)
    if len(set(found)) == 1:
        return found[0]
    return None


@register
class AnswerParser(Node):
    TYPE = "AnswerParser"
    TITLE = "Answer Parser"
    CATEGORY = "Evaluate"
    DESCRIPTION = """
Extracts the option letter. 'strict' accepts only a bare letter; failures are
logged (counted as wrong) instead of silently defaulting - Phase 5 defaulted
to 'A', which is available as on_failure='fallback A' for reproduction."""
    INPUTS = [Port("response", "RESPONSE"), Port("question", "QUESTION", optional=True)]
    OUTPUTS = [Port("answer", "ANSWER")]
    PARAMS = [
        Param("mode", "choice", "lenient", ["strict", "lenient", "answer-tag"]),
        Param("on_failure", "choice", "mark wrong", ["mark wrong", "fallback A"]),
    ]

    def run(self, ctx, inputs, p):
        r, q = inputs["response"], inputs.get("question")
        letters = tuple(sorted(q["options"])) if q else ("A", "B", "C", "D")
        text = (r.get("text") or "").strip()
        if r.get("probs"):
            letter = max(r["probs"], key=r["probs"].get)
        elif p["mode"] == "strict":
            m = re.fullmatch(r"\(?([A-Z])\)?\.?", text.upper())
            letter = m.group(1) if m and m.group(1) in letters else None
        elif p["mode"] == "answer-tag":
            tags = re.findall(r"ANSWER\s*[:：]?\s*\(?([A-Z])\b", text.upper())
            letter = tags[-1] if tags and tags[-1] in letters else parse_lenient(text, letters)
        else:
            letter = parse_lenient(text, letters)
            if letter is None and q:
                hits = [k for k, v in q["options"].items() if v.lower().rstrip(".") in text.lower()]
                letter = hits[0] if len(hits) == 1 else None
        parsed = letter is not None
        if not parsed and p["on_failure"] == "fallback A":
            letter = "A"
        ans = {"letter": letter, "parsed": parsed, "raw": text, "probs": r.get("probs"),
               "latency": r.get("latency"), "model": r.get("model")}
        return Result(outputs={"answer": ans},
                      ui={"metrics": {"letter": letter or "∅", "parsed": parsed}, "text": text[:400]})


# ----------------------------------------------------------------------------- routing
@register
class SelectiveGate(Node):
    TYPE = "SelectiveGate"
    TITLE = "Selective Gate (by subtask)"
    CATEGORY = "Prompt"
    DESCRIPTION = """
Phase 5's recommendation made executable: send the guided image/prompt only
for the listed subtasks and the plain ones otherwise."""
    INPUTS = [Port("question", "QUESTION"), Port("guided_image", "IMAGE"), Port("plain_image", "IMAGE"),
              Port("guided_prompt", "TEXT"), Port("plain_prompt", "TEXT")]
    OUTPUTS = [Port("image", "IMAGE"), Port("prompt", "TEXT")]
    PARAMS = [Param("guided_subtasks", "text", "Defect Classification, Defect Analysis",
                    help="comma-separated; options: " + ", ".join(SUBTASKS))]

    def run(self, ctx, inputs, p):
        wanted = {s.strip().lower() for s in p["guided_subtasks"].split(",") if s.strip()}
        guided = inputs["question"]["subtask"].lower() in wanted
        k = "guided" if guided else "plain"
        return Result(outputs={"image": inputs[f"{k}_image"], "prompt": inputs[f"{k}_prompt"]},
                      ui={"metrics": {"route": k}}, records=[{"kind": "context", "gate_route": k}])
