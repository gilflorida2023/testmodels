ollama_update
rm *.csv *.json 2>/dev/null
export repetitions=2
export INCLUDELIST=gemma,qwen
#export IGNORELIST=deepcoder,deepscaler,deepseek-r1:1.5b,deepseek-r1:7b,deepseek-r1:32b,deepseek-r1:14b,devstral:24b,granite3.3:2b-base,granite3.3:8b-base,llama3:latest,magistral:24b,minicpm-v:8b,phi4-mini-reasoning:3.8b,phi4-reasoning:latest,qwen2.5vl:3b,qwen2.5vl:7b,qwen2,qwen3:0.6b,qwq:latest
export IGNORELIST=
#python3 modelping.py "How many r's in 'strawberry'?" -s numeric -t 120  -r "$repetitions" -exp 3 -e "$IGNORELIST" -i "$INCLUDELIST"
#python3 modelping.py "How many a's in 'banana'?" -s numeric -t 120  -r "$repetitions" -exp 3 -e "$IGNORELIST" -i "$INCLUDELIST"
#python3 modelping.py "how many e's in 'elephant trunk'?" -s numeric -t 120 -exp 2 -r "$repetitions"  -e "$IGNORELIST" -i "$INCLUDELIST"
#python3 modelping.py "How many uppercase B's in 'BaBble'?" -s numeric -t 120 -exp 2 -r "$repetitions"  -e "$IGNORELIST" -i "$INCLUDELIST"
#python3 modelping.py "Count the number of words in 'The quick brown fox'?" -s numeric -t 120 -exp 4 -r "$repetitions" -e "$IGNORELIST" -i "$INCLUDELIST"
#python3 modelping.py "Count the number of vowels in 'audio'?" -s numeric -t 120 -exp 4 -r "$repetitions" -e "$IGNORELIST" -i "$INCLUDELIST"
#python3 modelping.py "Count the number of consonants in 'rhombus'?" -s numeric -t 120 -exp 5 -r "$repetitions" -e "$IGNORELIST" -i "$INCLUDELIST"
#python3 modelping.py "How many words in the sentence begins with r: 'The rooster runs fast'?" -s numeric -t 120 -exp 2 -r "$repetitions" -e "$IGNORELIST" -i "$INCLUDELIST"
#python3 modelping.py "How many letters in 'dog' + 'cat'? " -s numeric -t 180 -exp 6 -r "$repetitions"  -e "$IGNORELIST" -i "$INCLUDELIST"

python3 modelping.py "What is 9 × 5? Answer with a number." -s numeric -t 120 -exp 45 -r "$repetitions" -i "$INCLUDELIST"
python3 modelping.py "If 20 items are split equally among 5 people, how many does each get? Answer with a number." -s numeric -t 120 -exp 4 -r "$repetitions" -i "$INCLUDELIST"
python3 modelping.py "What is 15 + 8 + 7? Answer with a number." -s numeric -t 120 -exp 30 -r "$repetitions" -i "$INCLUDELIST"
python3 modelping.py "If x + 5 = 12, what is x? Answer with a number." -s numeric -t 120 -exp 7 -r "$repetitions" -i "$INCLUDELIST"
python3 modelping.py "If 3x = 18, what is x? Answer with a number." -s numeric -t 120 -exp 6 -r "$repetitions" -i "$INCLUDELIST"
python3 modelping.py "If 2x + 4 = 12, what is x? Answer with a number." -s numeric -t 180 -exp 4 -r "$repetitions" -i "$INCLUDELIST"
python3 modelping.py "How many prime numbers are between 10 and 20? Answer with a number." -s numeric -t 120 -exp 4 -r "$repetitions -i "$INCLUDELIST""
python3 modelping.py "How many factors does 12 have? Answer with a number." -s numeric -t 120 -exp 6 -r "$repetitions" -i "$INCLUDELIST"
python3 modelping.py "What is the smallest number divisible by both 4 and 6? Answer with a number." -s numeric -t 180 -exp 12 -r "$repetitions" -i "$INCLUDELIST"
python3 modelping.py "How many ways can you arrange 3 books on a shelf? Answer with a number." -s numeric -t 120 -exp 6 -r "$repetitions" -i "$INCLUDELIST"
python3 modelping.py "If you flip 2 coins, how many possible outcomes are there? Answer with a number." -s numeric -t 120 -exp 4 -r "$repetitions" -i "$INCLUDELIST"
python3 modelping.py "What is the next number in the sequence 2, 4, 8, 16? Answer with a number." -s numeric -t 120 -exp 32 -r "$repetitions" -i "$INCLUDELIST"
python3 modelping.py "How many planets in our solar system? Answer with a number." -s numeric -t 120 -exp 8 -r "$repetitions" -i "$INCLUDELIST"
python3 modelping.py "How many protons in a carbon atom? Answer with a number." -s numeric -t 120 -exp 6 -r "$repetitions" -i "$INCLUDELIST"
python3 modelping.py "What is the acceleration due to gravity on Earth in meters per second squared? Answer with a number." -s numeric -t 120 -exp 10 -r "$repetitions" -i "$INCLUDELIST"



