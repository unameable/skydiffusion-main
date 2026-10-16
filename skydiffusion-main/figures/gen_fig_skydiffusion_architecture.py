#!/usr/bin/env python3
"""Optional Gemini reproduction of the saved academic-plotting prompt.
The delivered candidates were generated with the built-in image tool because
GEMINI_API_KEY was not configured. This script retains the skill's Gemini path.
"""
from pathlib import Path
import os
import sys
import time

ROOT = Path(__file__).resolve().parent
MODEL = "gemini-3-pro-image-preview"

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY locally to use Gemini reproduction.")
        return 1
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    prompt = (ROOT / "prompt_skydiffusion_architecture.txt").read_text(encoding="utf-8")
    saved = []
    for attempt in range(1, 4):
        if attempt > 1:
            time.sleep(2)
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
            )
            for candidate in response.candidates or []:
                for part in candidate.content.parts or []:
                    if part.inline_data:
                        ext = ".jpg" if part.inline_data.mime_type == "image/jpeg" else ".png"
                        output = ROOT / f"fig_skydiffusion_gemini_attempt{attempt}{ext}"
                        output.write_bytes(part.inline_data.data)
                        saved.append(output)
                        print(f"Saved {output.name}")
                        break
                else:
                    continue
                break
        except Exception as error:
            # Exception details may contain provider request information; avoid printing them.
            print(f"Attempt {attempt} failed: {type(error).__name__}")
    print(f"Generated {len(saved)} candidates. Review before selecting.")
    return 0 if saved else 1

if __name__ == "__main__":
    sys.exit(main())
