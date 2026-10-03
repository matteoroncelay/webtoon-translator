"""Textes des requêtes envoyées au LLM, regroupés ici pour pouvoir les relire et
les améliorer sans toucher à la logique. Écrits en anglais : les modèles suivent
généralement mieux des consignes en anglais, même pour traduire vers le français."""

CLASSIFY = """You classify speech bubbles from a Korean webtoon. These bubbles have NO tail,
so the speaker is not visually indicated.

For each bubble, decide from the Korean text (OCR, may be noisy):
- "thought": internal monologue, a character thinking to themselves;
- "speech_offscreen": spoken words from a character who is not shown.
If unsure, choose the most likely one.

Return ONLY JSON: {{"bubbles": [{{"id": "...", "type": "thought" | "speech_offscreen"}}]}}

Bubbles:
{bubbles}
"""

CONTEXT = """You analyse dialogues from a Korean webtoon to find WHO speaks and TO WHOM.

Character profiles:
{profiles}

Previous lines (already analysed, for context only):
{previous}

For each bubble below:
- "known_speaker" was found from the drawing (tail direction): TRUST IT, only find "addressee".
- otherwise choose the speaker among "possible_speakers" if possible, using the tone
  (banmal / honorifics) and the profiles.
- for "thought" bubbles, addressee is "self".
Use EXACTLY the names from the profiles, or null if you cannot decide.

Return ONLY JSON: {{"bubbles": [{{"id": "...", "speaker": "...", "addressee": "..."}}]}}

Bubbles (in reading order):
{bubbles}
"""

TRANSLATE = """You are a professional webtoon translator, Korean to natural French.

Story so far:
{story}

Character profiles:
{profiles}

Translate every bubble of this page. Rules:
- natural, spoken French, faithful to the tone of each character;
- use "tu" or "vous" according to the speaker, the addressee and their relationship;
- keep translations short enough to fit in a speech bubble;
- narration boxes are written in a neutral narrative style.
Also write a 1-2 sentence summary of what happens on this page, for the next pages.

Return ONLY JSON:
{{"translations": [{{"id": "...", "text_fr": "..."}}], "page_summary": "..."}}

Bubbles of this page (in reading order):
{bubbles}
"""
