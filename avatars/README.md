# Character avatars

Drop an image here named after the character and it replaces the generated
placeholder automatically. No code change needed.

    avatars/Ana.png
    avatars/Alan.png
    avatars/Walter.png

Accepted extensions: `.png`, `.jpg`, `.jpeg`, `.webp`

**Recommended:** square, at least 500x500px. The UI crops to a square and
rounds the corners, so keep the face centred.

If no file matches a character name, the app serves a generated silhouette
whose colour is derived from the name — so every character still looks
distinct without any images present.

## Generating placeholders

Any image generator works. A prompt that matches the clinical tone:

> Photorealistic headshot portrait of a 35-year-old woman, neutral expression,
> plain light background, soft even lighting, shoulders visible, looking at
> camera. Documentary photography style.

Adjust age/gender per character — see `character_meta.json` for the roster.
Keep them consistent in framing and lighting so the set looks like one system.
