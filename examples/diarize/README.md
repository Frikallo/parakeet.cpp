# Speaker Diarization

Sortformer speaker diarization — identify who spoke when (up to 4 speakers).

## Build & Run

```bash
make build
./build/examples/example-diarize sortformer.safetensors meeting.wav
```

Requires Sortformer-117M weights.

## How It Works

Sortformer uses a NEST encoder (FastConformer) followed by a Transformer encoder to produce per-frame speaker activity probabilities, thresholded into segments.

## Expected Output

```
Detected 5 segment(s):

  Speaker 0: [0.56s - 2.96s]
  Speaker 0: [3.36s - 4.40s]
  Speaker 1: [4.80s - 6.24s]
  Speaker 0: [6.64s - 8.08s]
  Speaker 1: [8.48s - 10.24s]
```
