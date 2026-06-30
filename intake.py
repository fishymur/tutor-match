from anthropic import Anthropic

# Reads ANTHROPIC_API_KEY from enviroment
client = Anthropic()

# Uses claude haiku to frame query into academic topics used for embedding precisely
def rewrite_query(raw_query):
    prompt = (
        "A student described what they need help with, often vaguely or in plain English. "
        "Rewrite it as a short comma-separated list of the precise academic topics and "
        "keywords a tutor would list on their profile. "
        "Output ONLY the keywords — no preamble, no explanation.\n\n"
        f'Student request: "{raw_query}"'
    )
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()
