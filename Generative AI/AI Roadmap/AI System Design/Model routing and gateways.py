
"""
Model routing and gateways with real Azure OpenAI deployments, using LiteLLM.

Architecture:

    your app  ->  Gateway (this file)  ->  LiteLLM Router  ->  Azure OpenAI deployments

  Gateway: checks the Azure key is set, picks a route, tracks usage and cost.
  Router:  LiteLLM's helper that retries failed calls, falls back to another
           deployment, and stops using a deployment that keeps failing.

"""

import os
import sys
from pathlib import Path

os.environ.setdefault(
    "LITELLM_LOCAL_MODEL_COST_MAP", "True"
)  # bundled price list, no internet needed

import litellm
from dotenv import load_dotenv
from litellm import Router

litellm.suppress_debug_info = True
load_dotenv(
    Path(__file__).resolve().parents[1] / ".env"
)  # the .env lives in "Code examples"


def required(name):
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing {name} in the .env file")
    return value


ENDPOINT = required("AZURE_OPENAI_ENDPOINT")
API_KEY = required("AZURE_OPENAI_API_KEY")
API_VERSION = required("AZURE_OPENAI_API_VERSION")
DEAD_ENDPOINT = (
    "http://127.0.0.1:9"  # nothing listens here; used only to simulate an outage
)


# ---------------------------------------------------------------------------
# 1. Model list: logical tier names mapped to Azure deployments
# ---------------------------------------------------------------------------


TIERS = {
    "fast": required("AZURE_OPENAI_DEPLOYMENT"),  # gpt-5-mini: cheapest
    "mid": required("AZURE_OPENAI_DEPLOYMENT_54"),  # gpt-5.4
    "strong": required("AZURE_OPENAI_DEPLOYMENT_56sol"),  # gpt-5.6-sol: most careful
    "backup": required("AZURE_OPENAI_DEPLOYMENT_55"),  # gpt-5.5: used if "strong" fails
}


def build_model_list(outage=False):
    entries = []
    for tier, deployment in TIERS.items():
        params = {
            "model": f"azure/{deployment}",
            "api_key": API_KEY,
            "api_base": ENDPOINT,
            "api_version": API_VERSION,
        }
        if tier == "fast":
            params["rpm"] = (
                60  # requests per minute; the Router enforces this rate limit
            )
        if outage and tier == "strong":
            params["api_base"] = (
                DEAD_ENDPOINT  # simulated outage on the strong deployment
            )
        entries.append({"model_name": tier, "litellm_params": params})
    return entries


# ---------------------------------------------------------------------------
# 2. Routing: decide how a request should be handled
# ---------------------------------------------------------------------------
CLASSIFY_PROMPT = (
    "Classify the user's HR question. Reply with exactly one word:\n"
    "sensitive: termination, legal, performance, or dismissal issues\n"
    "policy: needs an answer from the HR policy\n"
    "easy: a general HR question that needs no policy"
)

# Made-up sample policies. The model can only cite what is given here.
POLICY_TEXT = (
    "HR-POLICY-2.1 Working hours: Standard working hours are 9:00 to 17:30, Monday to Friday. "
    "Flexible start times between 8:00 and 10:00 need manager approval.\n"
    "HR-POLICY-3.1 Parental leave: Eligible employees get 16 weeks of paid parental leave. "
    "Leave must start within 12 months of the child's birth or adoption.\n"
    "HR-POLICY-4.2 Annual leave: Employees get 25 paid leave days per year. "
    "Up to 5 unused annual leave days may be carried over to the next year. "
    "Carried-over days must be used by 31 March.\n"
    "HR-POLICY-5.1 Sick leave: Employees get 10 paid sick days per year. "
    "A doctor's note is required for sickness absence longer than 3 consecutive days.\n"
    "HR-POLICY-6.3 Remote work: Employees may work remotely up to 2 days per week "
    "with manager approval. Remote days must be logged in the HR portal.\n"
    "HR-POLICY-7.1 Expenses: Travel and meal expenses are reimbursed within 30 days "
    "of submission, with receipts attached. Alcohol is not reimbursable.\n"
    "HR-POLICY-8.2 Notice period: The notice period is 1 month during the first 2 years "
    "of service and 2 months after that. Unused leave is paid out on termination."
)
SYSTEM_PROMPT = (
    "You answer HR policy questions using only this policy text:\n"
    f"{POLICY_TEXT}\n"
    "Always cite the policy ID, such as HR-POLICY-4.2. If the text does not answer, say so."
)


def route(label):
    """Map the classifier's label to 'strong', 'fast', or 'cascade'."""
    return {"sensitive": "strong", "easy": "fast", "policy": "cascade"}.get(
        label, "cascade"  # unknown label: take the safe, checked path
    )


def has_citation(answer):
    """Quality check for the cascade: a good policy answer cites its policy."""
    return "HR-POLICY-" in answer


# ---------------------------------------------------------------------------
# 3. The gateway: auth, quota, usage tracking, and calls through the Router
# ---------------------------------------------------------------------------
class Gateway:
    def __init__(self, outage=False, token_quota=10_000):
        self.router = Router(
            model_list=build_model_list(outage),
            num_retries=2,  # retry transient errors
            fallbacks=[{"strong": ["backup"]}],  # if "strong" fails, try "backup"
            allowed_fails=2,  # after 2 failures...
            cooldown_time=30,  # ...stop using that deployment for 30 seconds
        )
        self.token_quota = token_quota
        self.tokens_used = 0  # total tokens used so far
        self.usage_log = []  # one record per model call

    def _call(self, tier, question, system=SYSTEM_PROMPT):
        if self.tokens_used >= self.token_quota:
            raise RuntimeError("the token quota is used up")

        response = self.router.completion(
            model=tier,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
        )
        tokens = response.usage.total_tokens
        cost = litellm.completion_cost(completion_response=response)
        self.tokens_used += tokens
        self.usage_log.append(
            {
                "tier": tier,
                "model": response.model,
                "tokens": tokens,
                "cost": cost,
            }
        )
        return response.choices[0].message.content or "", response.model

    @staticmethod
    def _result(text, model, route_note, human_review=False):
        return {
            "answer": text,
            "model": model,
            "route": route_note,
            "human_review": human_review,
        }

    def _classify(self, question):
        # The cheapest deployment labels the question. The label decides the route.
        text, _ = self._call("fast", question, system=CLASSIFY_PROMPT)
        return text.strip().lower()

    def answer(self, question):
        label = self._classify(question)
        plan = route(label)
        if plan == "cascade":
            # Try each tier in order. Stop at the first answer that passes the check.
            for tier in ["fast", "mid", "strong"]:
                text, model = self._call(tier, question)
                if has_citation(text):
                    return self._result(
                        text, model, f"'{label}' -> cascade, accepted at '{tier}'"
                    )
            # Even the strongest tier gave no citation, so a human must check it.
            return self._result(
                text,
                model,
                f"'{label}' -> cascade, no tier cited a policy",
                human_review=True,
            )

        tier = "strong" if plan == "strong" else "fast"
        text, model = self._call(tier, question)
        return self._result(
            text, model, f"'{label}' -> {plan}", human_review=(plan == "strong")
        )


# ---------------------------------------------------------------------------
# 4. Demo
# ---------------------------------------------------------------------------
def show(result):
    print(f"  route:        {result['route']}")
    print(f"  model:        {result['model']}")
    print(f"  human review: {result['human_review']}")
    print(f"  answer:       {result['answer']}")


def demo():
    gw = Gateway()

    print("Type a HR question and press Enter. Press Enter on an empty line to stop.")
    while True:
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        show(gw.answer(question))

    # print("\n=== Outage check: the 'strong' deployment is unreachable, so the Router falls back to 'backup' ===")
    # try:
    #     outage_question = input("Type a question to send during the outage (Enter to skip): ").strip()
    # except (EOFError, KeyboardInterrupt):
    #     outage_question = ""
    # if outage_question:
    #     show(Gateway(outage=True).answer(outage_question))

    # print("\n=== Guard rails ===")
    # small = Gateway()
    # small.answer("What does HR stand for?")
    # small.token_quota = 1  # the quota is now used up
    # try:
    #     small.answer("What does HR stand for?")
    # except RuntimeError as error:
    #     print("  quota     ->", error)

    print("\n=== Usage log (kept by the gateway) ===")
    for record in gw.usage_log:
        print(
            f"  {record['tier']:<7} {record['model']:<12} tokens={record['tokens']:<5} cost=${record['cost']:.6f}"
        )
    print(f"  total cost: ${sum(record['cost'] for record in gw.usage_log):.6f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(
        encoding="utf-8"
    )  # model answers can contain characters the Windows console cannot show
    demo()


# Sample questions to try:
# What does HR stand for?
# Can I carry over unused leave to next year?
# Can I carry over leave if my termination date is in March?
# What is the parental leave policy?
# How many paid sick days do I get per year?
# Can I work from home three days a week?
# What are the standard working hours?
# Can I claim alcohol on a business trip?
# How long is my notice period if I have worked here for 5 years?
# Is a doctor's note needed for a 2-day sickness absence?
# If I am fired tomorrow, do I still get paid for unused leave?
# How many annual leave days do I get?
# Can I start work at 8:30 instead of 9:00?
