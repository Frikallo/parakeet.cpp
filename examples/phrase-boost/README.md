# Phrase Boosting

Context biasing to improve recognition of domain-specific vocabulary (proper nouns, product names, jargon).

## Build & Run

```bash
make build
./build/examples/example-phrase-boost model.safetensors vocab.txt audio.wav
```

## How It Works

Phrase boosting builds a token-level trie from boost phrases and biases the decoder's log-probabilities toward trie-matched tokens during greedy decode. This steers recognition without retraining.

- `boost_phrases`: list of words/phrases to boost
- `boost_score`: log-probability bias strength (default: 5.0)

## Expected Output

```
Baseline: Well, I don't wish to see it anymore, observed Phoebi, turning away...
Boosted:  Well, I don't wish to see it anymore, observed Phoebe, turning away...
```
