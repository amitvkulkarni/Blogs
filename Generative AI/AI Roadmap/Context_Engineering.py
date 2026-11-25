import difflib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from openai import AzureOpenAI


@dataclass
class Settings:
    """Every tunable value lives here. Change it here, not inside the functions."""

    env_file: Path = Path(__file__).resolve().parents[1] / ".env"
    ref_dir: Path = Path(__file__).resolve().parent / "ref docs"
    doc_glob: str = "policy_*.txt"
    corpus_size: int = 300  # documents loaded from ref_dir
    demo_docs: int = 60  # documents used by most demos
    # Regex with named groups office, topic and code. Change it if your documents are worded differently.
    record_pattern: str = (
        r"the (?P<office>.+?) office approval code for (?P<topic>.+?) "
        r"requests is (?P<code>[A-Z]{2}-\d{4})"
    )
    needle_office: str = "Lisbon"  # not in ref docs, so the model must read the needle
    needle_topic: str = "travel"
    needle_code: str = "LX-7731"
    needle_template: str = (
        "Policy NEEDLE: the {office} office approval code for {topic} requests is {code}. "
        "Requests above the monthly limit go to the regional lead for review."
    )
    context_window_tokens: int = 32_000  # a teaching budget, not the model's real limit
    reserved_output_tokens: int = 1_000
    context_budget_tokens: int = 1_500  # budget for selected context
    max_age_days: int = 365
    dedupe_ratio: float = 0.97
    top_k: int = 3
    map_chunk_docs: int = 20
    keep_last_turns: int = 4
    cache_wait_seconds: int = 8
    tool_top_k: int = 3
    test_timeout_seconds: int = 20
    max_feature_attempts: int = 3
    reasoning_effort: str = (
        "minimal"  # keeps reasoning-token overhead low for these demos
    )
    workdir_prefix: str = "context_eng_"


# ---------------------------------------------------------------------------
# Azure client: the only place that talks to the API. It records every usage.
# ---------------------------------------------------------------------------


def required_env(name):
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing {name} in the .env file")
    return value


@dataclass
class Usage:
    prompt: int
    cached: int
    completion: int


class AzureLLM:
    """Azure OpenAI chat and embeddings. Every token count is what Azure reports."""

    def __init__(self, reasoning_effort):
        self.client = AzureOpenAI(
            azure_endpoint=required_env("AZURE_OPENAI_ENDPOINT"),
            api_key=required_env("AZURE_OPENAI_API_KEY"),
            api_version=required_env("AZURE_OPENAI_API_VERSION"),
        )
        self.chat_model = required_env("AZURE_OPENAI_DEPLOYMENT")
        self.embed_model = required_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        self.reasoning_effort = reasoning_effort
        self.totals = Counter()
        self._vectors = {}

    def chat(self, messages, max_tokens=2000, tools=None):
        kwargs = {
            "model": self.chat_model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "reasoning_effort": self.reasoning_effort,
        }
        if tools:
            kwargs["tools"] = tools
        resp = self.client.chat.completions.create(**kwargs)
        u = resp.usage
        details = u.prompt_tokens_details
        usage = Usage(
            u.prompt_tokens,
            (details.cached_tokens if details else 0) or 0,
            u.completion_tokens,
        )
        self.totals["calls"] += 1
        self.totals["prompt"] += usage.prompt
        self.totals["cached"] += usage.cached
        self.totals["completion"] += usage.completion
        return resp.choices[0].message, usage

    def ask(self, messages, max_tokens=2000):
        msg, usage = self.chat(messages, max_tokens)
        return (msg.content or "").strip(), usage

    def token_count(self, text):
        """Real token count of a text, read from the usage field of a tiny call."""
        _, usage = self.chat([user(text)], max_tokens=32)
        return usage.prompt

    def embed(self, texts, batch_size=64):
        """Embeddings, cached so each distinct text is sent to the API once."""
        missing = sorted({t for t in texts if t not in self._vectors})
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            resp = self.client.embeddings.create(model=self.embed_model, input=batch)
            for text, item in zip(batch, sorted(resp.data, key=lambda d: d.index)):
                self._vectors[text] = item.embedding
        return [self._vectors[t] for t in texts]

    def similarity(self, query, texts):
        """Cosine similarity between the query and each text."""
        vectors = self.embed([query] + texts)
        q, rest = vectors[0], vectors[1:]
        norm_q = math.sqrt(sum(x * x for x in q))
        return [
            sum(a * b for a, b in zip(q, v))
            / (norm_q * math.sqrt(sum(x * x for x in v)))
            for v in rest
        ]


# ---------------------------------------------------------------------------
# Messages, documents and records
# ---------------------------------------------------------------------------

SYSTEM = (
    "You are an HR policy assistant. Answer from the policy text, tool results, or facts the user "
    "stated earlier in this conversation. Use a tool when one can look up the answer. If none of "
    "these sources has the answer, say so instead of guessing."
)


def system(text):
    return {"role": "system", "content": text}


def user(text):
    return {"role": "user", "content": text}


@dataclass
class Document:
    source: str
    text: str
    age_days: int = 0


@dataclass(frozen=True)
class Record:
    office: str
    topic: str
    code: str


def joined(docs):
    return "\n".join(d.text for d in docs)


def load_corpus(settings):
    """Reads the policy .txt files from ref_dir and parses the approval codes they contain."""
    files = sorted(settings.ref_dir.glob(settings.doc_glob))[: settings.corpus_size]
    if len(files) < settings.corpus_size:
        raise SystemExit(
            f"Found {len(files)} files in {settings.ref_dir}, need {settings.corpus_size}. "
            "Run conect_engineeringva.py first."
        )
    now = datetime.now()
    docs = [
        Document(
            source=f.stem,
            text=f.read_text(encoding="utf-8").strip(),
            age_days=(now - datetime.fromtimestamp(f.stat().st_mtime)).days,
        )
        for f in files
    ]
    pattern = re.compile(settings.record_pattern)
    records = []
    for d in docs:
        m = pattern.search(d.text)
        if m:
            records.append(Record(**m.groupdict()))
    return docs, records


def insert_at_depth(docs, item, depth):
    """Place an item at a relative depth: 0.0 is the start, 1.0 is the end."""
    idx = round(depth * len(docs))
    return docs[:idx] + [item] + docs[idx:]


def question_for(record):
    return (
        f"What is the approval code for the {record.office} office for {record.topic} "
        "requests? Reply with the code only."
    )


def needle_document(settings):
    needle = settings.needle_template.format(
        office=settings.needle_office,
        topic=settings.needle_topic,
        code=settings.needle_code,
    )
    return Document(source="needle", text=needle)


def estimate_tokens(text):
    """Planning estimate only (about 4 characters per token). Use API counts for real numbers."""
    return max(1, len(text) // 4)


def dedupe_lines(text):
    counts = Counter(text.splitlines())
    seen, out = set(), []
    for line in text.splitlines():
        if line in seen:
            continue
        seen.add(line)
        out.append(line if counts[line] == 1 else f"{line} (x{counts[line]})")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Workspace: state that lives outside the context window
# ---------------------------------------------------------------------------


class Workspace:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, name, text):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def read(self, name, default=""):
        path = self.root / name
        return path.read_text(encoding="utf-8") if path.exists() else default

    def load_state(self, name):
        return json.loads(self.read(name, "{}"))

    def save_state(self, name, state):
        self.write(name, json.dumps(state, indent=2))

    def append_log(self, line):
        with open(self.root / "progress.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def recent_log(self, n=5):
        lines = self.read("progress.log").splitlines()
        return "\n".join(lines[-n:]) or "(no progress yet)"


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def banner(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def matches(answer, expected):
    if isinstance(expected, float):
        numbers = [
            float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*\.?\d*", answer)
        ]
        return any(abs(x - expected) < 0.01 for x in numbers)
    return expected.lower() in answer.lower()


def report(label, usage, answer=None, expected=None, expect_loss=False):
    cache = f"  cached={usage.cached}" if usage.cached else ""
    print(f"  {label:<36} prompt={usage.prompt:>6} tokens{cache}")
    if answer is not None:
        verdict = ""
        if expected is not None:
            if matches(answer, expected):
                verdict = "  [correct]"
            else:
                verdict = "  [expected loss]" if expect_loss else "  [WRONG]"
        print(f"      answer: {answer[:160]!r}{verdict}")


# ---------------------------------------------------------------------------
# Building blocks by concept
# ---------------------------------------------------------------------------


def select_context(llm, query, candidates, budget_tokens, max_age_days, dedupe_ratio):
    """Rank by relevance, then drop stale, duplicate and over-budget items. Returns (kept, log)."""
    scores = llm.similarity(query, [c.text for c in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda p: -p[0])
    kept, log, used = [], [], 0
    for _, c in ranked:
        if c.age_days > max_age_days:
            log.append(("drop-stale", c.source))
            continue
        if any(
            difflib.SequenceMatcher(None, c.text, k.text).ratio() >= dedupe_ratio
            for k in kept
        ):
            log.append(("drop-duplicate", c.source))
            continue
        cost = estimate_tokens(c.text)
        if used + cost > budget_tokens:
            log.append(("drop-budget", c.source))
            continue
        kept.append(c)
        used += cost
        log.append(("keep", c.source))
    return kept, log


def render_context(docs):
    """Tag every item with its source so the model can tell documents apart."""
    return "\n".join(
        f"[source: {d.source} | age: {d.age_days}d]\n{d.text}" for d in docs
    )


def turns_to_text(turns):
    return "\n".join(f"{role}: {text}" for role, text in turns)


def to_messages(turns):
    return [{"role": role, "content": text} for role, text in turns]


def compact_history(llm, turns, keep_last):
    """Summarise older turns with the model, and keep the most recent turns word for word."""
    older, recent = turns[:-keep_last], turns[-keep_last:]
    summary, usage = llm.ask(
        [
            system(
                "Summarize this conversation. Keep every code, number, name, and decision exactly as written."
            ),
            user(turns_to_text(older)),
        ],
        max_tokens=1500,
    )
    print(f"  summary call cost: prompt={usage.prompt} tokens")
    return [
        system(SYSTEM),
        user("Summary of earlier conversation:\n" + summary),
    ] + to_messages(recent)


def aggregate(rows, group_key, value_key):
    totals = Counter()
    for r in rows:
        totals[r[group_key]] += r[value_key]
    return {k: round(v, 2) for k, v in totals.items()}


def compact_rows(rows, group_key, value_key, ws, name="rows.json"):
    """Exact totals computed in code, plus a pointer to the full data saved on disk."""
    totals = aggregate(rows, group_key, value_key)
    largest = sorted(rows, key=lambda r: -r[value_key])[:3]
    path = ws.write(name, json.dumps(rows))
    return (
        f"Report: {len(rows)} rows. Exact totals by {group_key}: {json.dumps(totals)}. "
        f"Largest three: {json.dumps(largest)}. Full data saved at {path}."
    )


def order_for_attention(ranked):
    """Best items go to the two edges of the context; weaker items sit in the middle."""
    return ranked[0::2] + ranked[1::2][::-1]


def stuff(llm, docs, question):
    return llm.ask([system(SYSTEM), user(joined(docs) + "\n\n" + question)])


def retrieve(llm, docs, question, k):
    scores = llm.similarity(question, [d.text for d in docs])
    top = sorted(range(len(docs)), key=lambda i: -scores[i])[:k]
    return order_for_attention([docs[i] for i in top])


def map_reduce(llm, docs, question, chunk_size, instruction):
    """Each chunk gets its own call. The final call only sees the chunk notes."""
    notes, map_prompt = [], 0
    for start in range(0, len(docs), chunk_size):
        note, usage = llm.ask(
            [system(instruction), user(joined(docs[start : start + chunk_size]))],
            max_tokens=800,
        )
        notes.append(note)
        map_prompt += usage.prompt
    answer, usage = llm.ask(
        [
            system(SYSTEM),
            user("Summaries of each section:\n" + "\n".join(notes) + "\n\n" + question),
        ]
    )
    return answer, usage, notes, map_prompt


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # property name -> JSON type
    run: Callable[..., Any]

    def spec(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {p: {"type": t} for p, t in self.parameters.items()},
                    "required": list(self.parameters),
                },
            },
        }


def select_tools(llm, query, tools, k):
    """Expose only the k tools whose descriptions are closest to the query."""
    scores = llm.similarity(query, [f"{t.name}: {t.description}" for t in tools])
    ranked = sorted(zip(scores, tools), key=lambda p: -p[0])
    return [t for _, t in ranked[:k]]


@dataclass
class AgentRun:
    answer: str
    first_usage: Usage
    tools_called: list


def execute_tool(tool, arguments_json):
    try:
        return tool.run(**json.loads(arguments_json))
    except Exception as exc:  # the model sees the error and can retry
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_agent(llm, query, tools, max_steps=4):
    """The tool loop: ask, run the requested tools, send results back, repeat."""
    by_name = {t.name: t for t in tools}
    specs = [t.spec() for t in tools]
    messages = [system(SYSTEM), user(query)]
    first_usage, called = None, []
    for _ in range(max_steps):
        msg, usage = llm.chat(messages, tools=specs)
        first_usage = first_usage or usage
        if not msg.tool_calls:
            return AgentRun((msg.content or "").strip(), first_usage, called)
        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )
        for tc in msg.tool_calls:
            called.append(tc.function.name)
            result = execute_tool(by_name[tc.function.name], tc.function.arguments)
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)}
            )
    return AgentRun("", first_usage, called)


def run_subagent(llm, task, context_text):
    """A sub-agent works in its own context window and returns one sentence."""
    return llm.ask(
        [
            system("Answer from the text only, in one sentence."),
            user(context_text + "\n\nTask: " + task),
        ],
        max_tokens=800,
    )


@dataclass
class Feature:
    id: str
    description: str
    tests: str


def extract_code(text):
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return blocks[-1] if blocks else text


def run_tests(code, tests, timeout):
    """Runs the module against the tests in a temporary folder, with a timeout."""
    with tempfile.TemporaryDirectory() as d:
        Path(d, "solution.py").write_text(code, encoding="utf-8")
        Path(d, "test_solution.py").write_text(
            "from datetime import date\nfrom solution import *\n"
            + tests
            + "\nprint('ALL TESTS PASSED')\n",
            encoding="utf-8",
        )
        try:
            r = subprocess.run(
                [sys.executable, "test_solution.py"],
                cwd=d,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, f"timed out after {timeout} seconds"
        return r.returncode == 0, (r.stdout + r.stderr)[-800:]


SESSION_SYSTEM = (
    "You write one Python module. Reply with the complete module in a single ```python block "
    "and no other prose."
)


def run_feature(llm, ws, session_no, feature, settings):
    """A fresh session. Its only memory is the progress log and the feature spec."""
    messages = [
        system(SESSION_SYSTEM),
        user(
            f"Feature {feature.id}: {feature.description}\n\n"
            f"Progress notes from earlier sessions:\n{ws.recent_log()}\n\n"
            f"The module must pass these tests:\n```python\n{feature.tests}\n```"
        ),
    ]
    for attempt in range(1, settings.max_feature_attempts + 1):
        msg, usage = llm.chat(messages, max_tokens=4000)
        reply = msg.content or ""
        code = extract_code(reply)
        ok, output = run_tests(code, feature.tests, settings.test_timeout_seconds)
        print(
            f"  session {session_no}, attempt {attempt}: prompt={usage.prompt} tokens, "
            f"tests {'PASS' if ok else 'FAIL'}"
        )
        if ok:
            ws.write(f"{feature.id}.py", code)
            ws.append_log(
                f"{feature.id} passed in session {session_no} after {attempt} attempt(s)"
            )
            return True
        messages.append({"role": "assistant", "content": reply})
        messages.append(
            user(
                f"The tests failed:\n{output}\nReturn the full corrected module in one ```python block."
            )
        )
    ws.append_log(f"{feature.id} FAILED after {settings.max_feature_attempts} attempts")
    return False


# ---------------------------------------------------------------------------
# Demo wiring: the context shared by the demos, and the demo inputs
# ---------------------------------------------------------------------------


@dataclass
class Context:
    settings: Settings
    llm: AzureLLM
    docs: list
    records: list
    needle: Record
    workspace: Workspace

    def demo_docs(self, depth=None):
        """The first demo_docs documents, with the needle inserted at depth if given."""
        docs = self.docs[: self.settings.demo_docs]
        if depth is None:
            return docs
        return insert_at_depth(docs, needle_document(self.settings), depth)


NEEDLE_DEPTHS = [("start", 0.0), ("middle", 0.5), ("end", 1.0)]
COST_CENTRE = "CC-5521"
HISTORY = [
    (
        "user",
        f"My cost centre code is {COST_CENTRE} for the Berlin team. Please remember it for expense claims.",
    ),
    ("assistant", f"Noted. I will use {COST_CENTRE} for expense claims."),
] + [
    pair
    for topic in [
        "remote work",
        "laptop refresh",
        "training budget",
        "parental leave",
        "office keys",
    ]
    for pair in [
        ("user", f"What is the policy on {topic}? Give me the short version."),
        (
            "assistant",
            f"Short version: {topic} follows the standard HR guideline; ask your manager for exceptions.",
        ),
    ]
]
QUESTION_COST = (
    "Which cost centre should my travel claim use? Reply with the code only."
)
MAP_INSTRUCTION = (
    "List every office approval code in this text as 'Office: CODE', one per line, copying the "
    "office name and code exactly. If there are none, reply NONE."
)
TOOL_QUESTION = "What is the approval code for the Lisbon office for travel requests? Reply with the code only."
EXPENSE_QUESTION = (
    "What is the total amount spent on Travel? Reply with the number only."
)
FEATURES = [
    Feature(
        "F1",
        "count_business_days(start, end): number of Monday-to-Friday days between two datetime.date "
        "values, inclusive of both ends. Return an int.",
        "assert count_business_days(date(2024, 1, 1), date(2024, 1, 5)) == 5\n"
        "assert count_business_days(date(2024, 1, 6), date(2024, 1, 7)) == 0\n"
        "assert count_business_days(date(2024, 1, 5), date(2024, 1, 8)) == 2",
    ),
    Feature(
        "F2",
        "leave_balance(entitlement_days, used_days, carry_over_days=0): remaining leave days as a "
        "float, never negative.",
        "assert leave_balance(20, 6) == 14\n"
        "assert leave_balance(20, 6, 9) == 19\n"
        "assert leave_balance(10, 15) == 0",
    ),
]


def make_expense_rows(n=240, seed=3):
    """Demo data: a fixed expense report, so the numbers repeat between runs."""
    rng = random.Random(seed)
    categories = ["Travel", "Meals", "Software", "Equipment", "Training"]
    return [
        {
            "id": f"EXP-{i:05d}",
            "category": rng.choice(categories),
            "amount": round(rng.uniform(8, 900), 2),
        }
        for i in range(n)
    ]


def build_tools(ctx):
    docs = ctx.demo_docs(depth=0.5)
    records = ctx.records + [ctx.needle]

    def search_policy(topic):
        scores = ctx.llm.similarity(topic, [d.text for d in docs])
        best = sorted(range(len(docs)), key=lambda i: -scores[i])[:2]
        return {"matches": [docs[i].text for i in best]}

    def lookup_approval_code(office, topic):
        for r in records:
            same_office = r.office.lower() == office.lower()
            same_topic = (
                r.topic.lower() in topic.lower() or topic.lower() in r.topic.lower()
            )
            if same_office and same_topic:
                return {"code": r.code}
        return {"error": f"no approval code for {office} / {topic}"}

    def not_connected(**_):
        return {"error": "this system is not connected in the demo"}

    return [
        Tool(
            "search_policy",
            "Search the HR policy documents by topic and return the closest matches.",
            {"topic": "string"},
            search_policy,
        ),
        Tool(
            "lookup_approval_code",
            "Return the approval code for an office and a request topic.",
            {"office": "string", "topic": "string"},
            lookup_approval_code,
        ),
        Tool(
            "get_leave_balance",
            "Return remaining annual leave days for an employee id.",
            {"employee_id": "string"},
            not_connected,
        ),
        Tool(
            "book_travel",
            "Request a business trip to a city on a date (YYYY-MM-DD).",
            {"city": "string", "date": "string"},
            not_connected,
        ),
        Tool(
            "submit_expense",
            "Submit an expense claim with an amount and a category.",
            {"amount": "number", "category": "string"},
            not_connected,
        ),
    ]


# ---------------------------------------------------------------------------
# The eight demos
# ---------------------------------------------------------------------------


def demo_budget_and_rot(ctx):
    banner("1. Context as a budget: measure real tokens, then test lost-in-the-middle")
    s, llm = ctx.settings, ctx.llm
    parts = [
        ("system instructions", SYSTEM),
        (f"{len(ctx.docs)} policy docs", joined(ctx.docs)),
        (f"conversation ({len(HISTORY)} turns)", turns_to_text(HISTORY)),
    ]
    window, reserved = s.context_window_tokens, s.reserved_output_tokens
    print(f"  Budget: window {window} tokens, reserved for output {reserved}\n")
    used = 0
    for label, text in parts:
        tokens = llm.token_count(text)
        used += tokens
        print(f"  {label:<26} {tokens:>7} tokens  ({tokens / window:6.1%} of window)")
    print(f"  {'free for everything else':<26} {window - reserved - used:>7} tokens")

    print("\n  Needle test: the same question, with the needle at different depths")
    question = question_for(ctx.needle)
    for label, depth in NEEDLE_DEPTHS:
        text = joined(ctx.demo_docs(depth=depth)) + "\n\n" + question
        ans, usage = llm.ask([system(SYSTEM), user(text)])
        report(f"needle at {label} of demo docs", usage, ans, expected=ctx.needle.code)


def demo_selection(ctx):
    banner("2. Selection & assembly: keep, retrieve, or drop")
    s, llm = ctx.settings, ctx.llm
    question = question_for(ctx.needle)
    candidates = ctx.demo_docs(depth=0.5)
    candidates.append(
        Document(
            "policy-OLD-2019",
            f"Policy OLD-2019: the {ctx.needle.office} office approval code for {ctx.needle.topic} requests was LX-1111. Superseded in 2024.",
            age_days=900,
        )
    )
    candidates += [
        Document(f"copy-of-{d.source}", d.text, age_days=200) for d in ctx.docs[:3]
    ]

    kept, log = select_context(
        llm,
        question,
        candidates,
        s.context_budget_tokens,
        s.max_age_days,
        s.dedupe_ratio,
    )
    print(
        f"  {len(candidates)} candidates -> decisions: {dict(Counter(a for a, _ in log))}"
    )
    print(f"  Kept sources: {[d.source for d in kept]}\n")

    ans, usage = llm.ask([system(SYSTEM), user(joined(candidates) + "\n\n" + question)])
    report("dump everything", usage, ans, expected=ctx.needle.code)
    ans, usage = llm.ask(
        [system(SYSTEM), user(render_context(kept) + "\n\n" + question)]
    )
    report("assembled context", usage, ans, expected=ctx.needle.code)


def demo_compression(ctx):
    banner("3. Compression & compaction: reduce size without losing task facts")
    s, llm = ctx.settings, ctx.llm
    ans, usage = llm.ask(
        [system(SYSTEM)] + to_messages(HISTORY) + [user(QUESTION_COST)]
    )
    report("A. full history", usage, ans, expected=COST_CENTRE)

    summarized = compact_history(llm, HISTORY, s.keep_last_turns) + [
        user(QUESTION_COST)
    ]
    ans, usage = llm.ask(summarized)
    report("B. summarized older turns", usage, ans, expected=COST_CENTRE)

    ans, usage = llm.ask(
        [system(SYSTEM)]
        + to_messages(HISTORY[-s.keep_last_turns :])
        + [user(QUESTION_COST)]
    )
    report(
        "C. keep only the last turns",
        usage,
        ans,
        expected=COST_CENTRE,
        expect_loss=True,
    )

    print("\n  Tool output compaction: expense report rows")
    rows = make_expense_rows()
    exact = aggregate(rows, "category", "amount")["Travel"]
    ans, usage = llm.ask(
        [
            system(SYSTEM),
            user(f"Expense report JSON:\n{json.dumps(rows)}\n\n{EXPENSE_QUESTION}"),
        ]
    )
    report("raw tool output", usage, ans, expected=exact, expect_loss=True)
    compact = compact_rows(rows, "category", "amount", ctx.workspace, "expenses.json")
    ans, usage = llm.ask([system(SYSTEM), user(f"{compact}\n\n{EXPENSE_QUESTION}")])
    report("compacted tool output", usage, ans, expected=exact)
    print(f"      exact total computed in code: {exact}")


def demo_offloading(ctx):
    banner("4. Memory & offloading: state in a file vs a growing history")
    llm, ws = ctx.llm, ctx.workspace
    reference = user("Policy reference:\n" + joined(ctx.demo_docs(depth=0.5)))
    first, last = ctx.records[0], ctx.records[-1]
    steps = [
        ("needle office code", question_for(ctx.needle), ctx.needle.code),
        (f"{first.office} code", question_for(first), first.code),
        (f"{last.office} code", question_for(last), last.code),
        ("Receipt retention", "How long must receipts be kept?", "90 days"),
    ]

    print("  A. Growing history: every step resends all earlier questions and answers")
    history = [system(SYSTEM), reference]
    naive_total = 0
    for name, question, expected in steps:
        history.append(user(question))
        ans, usage = llm.ask(history, max_tokens=1500)
        history.append({"role": "assistant", "content": ans})
        naive_total += usage.prompt
        report(name, usage, ans, expected=expected)

    print("\n  B. Offloaded state: each step reads a state file, not the whole history")
    ws.save_state("findings.json", {})
    offloaded_total = 0
    for name, question, expected in steps:
        findings = ws.read("findings.json", "{}")
        messages = [
            system(SYSTEM),
            reference,
            user("Findings so far (from the state file):\n" + findings),
            user(question),
        ]
        ans, usage = llm.ask(messages, max_tokens=1500)
        offloaded_total += usage.prompt
        report(name, usage, ans, expected=expected)
        state = ws.load_state("findings.json")
        state[name] = ans
        ws.save_state("findings.json", state)

    print("\n  C. A fresh session resumes from the file alone")
    ans, usage = llm.ask(
        [
            system(SYSTEM),
            user("Findings so far:\n" + ws.read("findings.json")),
            user("List the names of the findings already recorded."),
        ],
        max_tokens=800,
    )
    report("fresh session, resume", usage, ans)
    print(
        f"\n  Total prompt tokens over {len(steps)} steps: growing history={naive_total}, offloaded state={offloaded_total}"
    )


def demo_long_context(ctx):
    banner("5. Long context: stuffing vs retrieval vs map-reduce")
    s, llm = ctx.settings, ctx.llm
    docs = ctx.demo_docs(depth=0.6)
    question = question_for(ctx.needle)

    ans, usage = stuff(llm, docs, question)
    report(f"A. stuffing (all {len(docs)} docs)", usage, ans, expected=ctx.needle.code)

    top = retrieve(llm, docs, question, s.top_k)
    ans, usage = llm.ask([system(SYSTEM), user(joined(top) + "\n\n" + question)])
    report(
        f"B. retrieval, top {s.top_k}, edge order", usage, ans, expected=ctx.needle.code
    )

    ans, usage, notes, map_prompt = map_reduce(
        llm, docs, question, s.map_chunk_docs, MAP_INSTRUCTION
    )
    report(
        f"C. map-reduce ({len(notes)} chunk calls + 1)",
        usage,
        ans,
        expected=ctx.needle.code,
    )
    print(f"      map step prompt tokens across chunks: {map_prompt}")
    for i, note in enumerate(notes):
        print(f"      chunk {i + 1} note: {note.strip()[:90]!r}")


def demo_tools(ctx):
    banner("6. Tool context: expose only relevant tools, compact large results")
    s, llm = ctx.settings, ctx.llm
    tools = build_tools(ctx)
    all_specs = [t.spec() for t in tools]
    msg, usage = llm.chat([system(SYSTEM), user(TOOL_QUESTION)], tools=all_specs)
    report(f"A. all {len(tools)} tool definitions", usage)
    print(f"      model chose: {[tc.function.name for tc in msg.tool_calls or []]}")

    chosen = select_tools(llm, TOOL_QUESTION, tools, s.tool_top_k)
    print(f"      pre-selected {len(chosen)}: {[t.name for t in chosen]}")
    run = run_agent(llm, TOOL_QUESTION, chosen)
    report(
        f"B. selected {len(chosen)} tool definitions",
        run.first_usage,
        run.answer,
        expected=ctx.needle.code,
    )
    print(f"      tools actually called: {run.tools_called}")

    print(
        "\n  Large tool result: the same question with the raw report and with a compacted report"
    )
    rows = make_expense_rows()
    exact = aggregate(rows, "category", "amount")["Travel"]
    ans, usage = llm.ask(
        [
            system(SYSTEM),
            user(f"Tool result (raw):\n{json.dumps(rows)}\n\n{EXPENSE_QUESTION}"),
        ]
    )
    report("raw tool result", usage, ans, expected=exact, expect_loss=True)
    compact = compact_rows(rows, "category", "amount", ctx.workspace, "tool_rows.json")
    ans, usage = llm.ask(
        [
            system(SYSTEM),
            user(f"Tool result (compacted):\n{compact}\n\n{EXPENSE_QUESTION}"),
        ]
    )
    report("compacted tool result", usage, ans, expected=exact)


def demo_caching(ctx):
    banner("7. Caching: a stable prefix gets cache hits, a changing prefix does not")
    s, llm = ctx.settings, ctx.llm
    static = SYSTEM + "\n\n" + joined(ctx.demo_docs(depth=0.5))
    first = ctx.records[0]
    q1, q2 = question_for(first), question_for(ctx.needle)

    ans, usage = llm.ask([system(static), user(q1)])
    report("1. stable prefix, first call", usage, ans, expected=first.code)
    time.sleep(s.cache_wait_seconds)
    ans, usage = llm.ask([system(static), user(q2)])
    report("2. same prefix, second call", usage, ans, expected=ctx.needle.code)
    stable_rate = usage.cached / usage.prompt if usage.prompt else 0

    volatile = f"Current time: {datetime.now().isoformat()}\n\n" + static
    ans, usage = llm.ask([system(volatile), user(q2)])
    report("3. timestamp at the top", usage, ans, expected=ctx.needle.code)
    volatile_rate = usage.cached / usage.prompt if usage.prompt else 0
    print(
        f"\n  Cache utilization: stable call {stable_rate:.0%}; volatile call {volatile_rate:.0%}"
    )

    log = "\n".join(["Retrying request to HR portal..."] * 30 + ["Connected."])
    print(
        f"\n  Dedupe repeated log lines before they enter the context: {len(log)} -> {len(dedupe_lines(log))} characters"
    )


def demo_harness(ctx):
    banner("8. Harness: sub-agent for a lookup, fresh sessions, self-run tests")
    s, llm, ws = ctx.settings, ctx.llm, ctx.workspace
    target = ctx.records[0]
    finding, usage = run_subagent(
        llm,
        f"What is the approval code for the {target.office} office for {target.topic} requests?",
        joined(ctx.demo_docs()),
    )
    report("sub-agent read the demo docs", usage, finding, expected=target.code)
    ws.append_log(f"Policy finding (sub-agent): {finding}")

    for session_no, feature in enumerate(FEATURES, start=1):
        run_feature(llm, ws, session_no, feature, s)

    print("\n  Progress file (the durable record):")
    for line in ws.read("progress.log").splitlines():
        print(f"    {line}")


SECTIONS = {
    "1": demo_budget_and_rot,
    "2": demo_selection,
    "3": demo_compression,
    "4": demo_offloading,
    "5": demo_long_context,
    "6": demo_tools,
    "7": demo_caching,
    "8": demo_harness,
}


def main():
    sys.stdout.reconfigure(errors="replace")
    settings = Settings()
    load_dotenv(settings.env_file)
    chosen = sys.argv[1:] or list(SECTIONS)
    for key in chosen:
        if key not in SECTIONS:
            raise SystemExit(f"Unknown section {key}; choose from 1 to 8")

    llm = AzureLLM(settings.reasoning_effort)
    docs, records = load_corpus(settings)
    needle = Record(settings.needle_office, settings.needle_topic, settings.needle_code)
    workdir = Path(tempfile.mkdtemp(prefix=settings.workdir_prefix))
    try:
        ctx = Context(settings, llm, docs, records, needle, Workspace(workdir))
        for key in chosen:
            SECTIONS[key](ctx)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    t = llm.totals
    print(
        f"\nTotal: {t['calls']} model calls, prompt={t['prompt']} tokens "
        f"(cached {t['cached']}), completion={t['completion']} tokens"
    )


if __name__ == "__main__":
    main()
