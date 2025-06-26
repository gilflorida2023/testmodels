ollama_update
export IGNORELIST=deepcoder:14b,deepcoder,deepscaler,deepseek-r1:1.5b,deepseek-r1:7b,deepseek-r1:32b,deepseek-r1:14b,devstral:24b,granite3.3:2b-base,granite3.3:8b-base,llama3:latest,llava:7b,abliterated,magistral:24b,minicpm-v:8b,phi4-mini-reasoning:3.8b,phi4-reasoning:latest,qwen2.5vl:3b,qwen2.5vl:7b,qwen2,qwen3:0.6b,qwq:latest
python3 modelping.py "How many 'r’s in 'strawberry'?" -s numeric -t 120  -r 1 -exp 3 -e "$IGNORELIST"
python3 modelping.py "How many 'a’s in 'banana'?" -s numeric -t 120  -r 1 -exp 3 -e "$IGNORELIST"
python3 modelping.py "how many e's in 'elephant trunk'?" -s numeric -t 120 -exp 2 -r 1  -e "$IGNORELIST"
python3 modelping.py "How many uppercase 'B’s in 'BaBble'?" -s numeric -t 120 -exp 2 -r 1  -e "$IGNORELIST"
python3 modelping.py "Count the number of words in 'The quick brown fox'?" -s numeric -t 120 -exp 4 -r 1 -e "$IGNORELIST"
python3 modelping.py "Count the number of vowels in 'audio'?" -s numeric -t 120 -exp 4 -r 1 -e "$IGNORELIST"
python3 modelping.py "Count the number of consonants in 'rhombus'?" -s numeric -t 120 -exp 5 -r 1 -e "$IGNORELIST"
python3 modelping.py "How many words in the sentence 'The cat runs fast'?" -s numeric -t 120 -exp 1 -r 1 -e "$IGNORELIST"
python3 modelping.py "How many letters in 'dog' + 'cat'? " -s numeric -t 180 -exp 6 -r 1  -e "$IGNORELIST"
