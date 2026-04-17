import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from   router import detect_intent

tests = [
    "My father has a headache",
    "My father suddenly has chest pain and is sweating heavily",
    "I burned my hand with hot tea",
    "میرے والد کو سینے میں درد ہے",
    "اگر کسی کو سر درد ہو تو کیا کرنا چاہیے؟",
    "mere father ko chest pain ho raha hai"
]

for q in tests:
    print("\nQ:", q)
    print(detect_intent(q))