from anthropic import Anthropic

# Reads ANTHROPIC_API_KEY from your environment automatically — the key is never in this file.
client = Anthropic()

def rewrite_query(raw_query):
    """Expand a vague, student-phrased request into precise topic keywords."""
    prompt = (
        "A student described what they need help with, often vaguely or in plain English. "
        "Rewrite it as a short comma-separated list of the precise academic topics and "
        "keywords a tutor would list on their profile. "
        "Output ONLY the keywords — no preamble, no explanation.\n\n"
        f'Student request: "{raw_query}"'
    )
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=80,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()

# This block runs ONLY when you run this file directly (not when another file imports it).
if __name__ == "__main__":
    samples = [
        "how do I show my code stays fast as the input grows huge",
        "what stays the same when you rotate or reflect a shape",
    ]
    for q in samples:
        print(f"\nraw:     {q}")
        print(f"rewrite: {rewrite_query(q)}")