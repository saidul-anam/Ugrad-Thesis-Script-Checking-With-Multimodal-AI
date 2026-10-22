#!/usr/bin/env python3
"""
Generate sample PDF scripts for local development and offline mock pipeline testing.
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def create_sample_pdf(output_path: str, title: str, sample_text: list):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create white page (A4-ish aspect ratio: 800 x 1130)
    img = Image.new("RGB", (800, 1130), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title header
    draw.rectangle([(40, 40), (760, 100)], fill=(240, 240, 240), outline=(180, 180, 180))
    draw.text((60, 60), f"Sample Exam Script: {title}", fill=(0, 0, 0))

    # Ruled lines and sample handwriting text
    y = 150
    for line in sample_text:
        # Draw faint ruled line
        draw.line([(40, y + 25), (760, y + 25)], fill=(220, 220, 240), width=1)
        # Draw simulated text
        draw.text((60, y), line, fill=(30, 30, 80))
        y += 45

    img.save(output_path, "PDF", resolution=100.0)
    print(f"[Sample Generator] Created: {output_path}")


def main():
    # 1. Bangla sample
    bangla_text = [
        "প্রশ্ন ১: উদ্দীপকে বর্ণিত ঘটনাটির সাথে 'অপরিচিতা' গল্পের মিল আলোচনা কর।",
        "উত্তর:",
        "উদ্দীপকে বর্ণিত চরিত্রটির মধ্যে সামাজিক যৌতুক প্রথার বিরুদ্ধে প্রতিবাদী মনোভাব দেখা যায়।",
        "অনুরূপভাবে রবীন্দ্রনাথ ঠাকুরের 'অপরিচিতা' গল্পে অনুপমের মামার লোভ ও যৌতুক লিপ্সার",
        "বিপরীতে কল্যাণীর পিতা শম্ভুনাথ সেনের দৃঢ় আত্মমর্যাদাবোধ প্রকাশিত হয়েছে।",
        "কল্যাণী নিজে অন্যায়ের বিরুদ্ধে দাঁড়িয়ে বিয়ে প্রত্যাখ্যান করেছিল।",
        "অতএব, উদ্দীপকের মূল ভাবের সাথে গল্পের আত্মমর্যাদাবোধ ও প্রতিবাদের দিকটি সংগতিপূর্ণ।"
    ]
    create_sample_pdf("data/raw_pdfs/bangla/sample_bangla_01.pdf", "Bangla Creative Question (সৃজনশীল)", bangla_text)
    create_sample_pdf("data/samples/sample_bangla_01.pdf", "Bangla Creative Question (সৃজনশীল)", bangla_text)

    # 2. English sample
    english_text = [
        "Question 1: Write an essay on the Impact of Artificial Intelligence on Education.",
        "Answer:",
        "Artificial intelligence is transforming modern education by enabling personalized learning.",
        "Students can learn at their own pace with intelligent tutoring systems.",
        "However, excessive reliance on AI may reduce critical thinking and problem-solving skills.",
        "In conclusion, AI should be used as a supportive tool alongside human educators."
    ]
    create_sample_pdf("data/raw_pdfs/english/sample_english_01.pdf", "English Essay Writing", english_text)
    create_sample_pdf("data/samples/sample_english_01.pdf", "English Essay Writing", english_text)


if __name__ == "__main__":
    main()
