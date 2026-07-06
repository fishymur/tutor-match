from anthropic import Anthropic

client = Anthropic()

#Prevents recalling same query
_rewrite_cache = {}

def rewrite_query(raw_query):
    """Expand a vague, student-phrased request into precise topic keywords."""
    key = raw_query.strip().lower()
    if key in _rewrite_cache:
        return _rewrite_cache[key]

    prompt = (
        "A student described what they need help with, often vaguely or in plain English. "
        "Rewrite it as a short comma-separated list of the precise academic topics and "
        "keywords a tutor would list on their profile as thier specialties. "
        "Output ONLY the keywords — no preamble, no explanation.\n\n"
        f'Student request: "{raw_query}"'
    )
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    result = message.content[0].text.strip()
    _rewrite_cache[key] = result
    return result