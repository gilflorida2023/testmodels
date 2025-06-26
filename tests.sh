ollama_update
export IGNORELIST=deepcoder:14b,deepcoder,deepscaler,deepseek-r1:1.5b,deepseek-r1:7b,deepseek-r1:32b,deepseek-r1:14b,devstral:24b,granite3.3:2b-base,granite3.3:8b-base,llama3:latest,llava:7b,abliterated,magistral:24b,minicpm-v:8b,phi4-mini-reasoning:3.8b,phi4-reasoning:latest,qwen2.5vl:3b,qwen2.5vl:7b,qwen2,qwen3:0.6b,qwq:latest
python3 modelping.py "Count the number of 'r’s in 'strawberry'. Return only the number, nothing else—no text, no explanations, no spaces, no punctuation." -t 120  -r 1 -exp 3 -e "$IGNORELIST"
python3 modelping.py "Count the number of 'a’s in 'banana'. Return only the number, nothing else—no text, no explanations, no spaces, no punctuation." -t 120  -r 1 -exp 3 -e "$IGNORELIST"
python3 modelping.py "Return only the number: 0..100 that represents the how many e's in 'elephant trunk'." -t 120 -exp 2 -r 1  -e "$IGNORELIST"
python3 modelping.py "Return only the number: 0..6 representing count of uppercase 'B’s in 'BaBble'." -t 120 -exp 2 -r 1  -e "$IGNORELIST"
python3 modelping.py "Count the number of words in 'The quick brown fox'. Return numeric answer : 0 <= answer <= 8." -t 120 -exp 4 -r 1 -e "$IGNORELIST"
python3 modelping.py "Count the number of vowels in 'audio'. Return only numeric answer: 0<=answer<=5." -t 120 -exp 4 -r 1 -e "$IGNORELIST"
python3 modelping.py "Count the number of consonants in 'rhombus'. Return only the number." -t 120 -exp 5 -r 1 -e "$IGNORELIST"
python3 modelping.py "In the sentence 'The cat runs fast', count the number of words that start with 'r'. Return only the number." -t 120 -exp 1 -r 1 -e "$IGNORELIST"
python3 modelping.py "How many letters in 'dog' + 'cat'? Return only the number:0<=answer<=7." -t 180 -exp 6 -r 1  -e "$IGNORELIST"
