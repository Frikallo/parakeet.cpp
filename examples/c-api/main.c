/* Pure C99 usage of parakeet.cpp via the C FFI
 *
 * Usage: example-c-api <model.safetensors> <vocab.txt> <audio.wav>
 */

#include <parakeet/api/parakeet_c.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr,
                "Usage: %s <model.safetensors> <vocab.txt> <audio.wav>\n",
                argv[0]);
        return 1;
    }

    const char *weights = argv[1];
    const char *vocab = argv[2];
    const char *audio_path = argv[3];

    printf("parakeet.cpp %s (C API)\n\n", parakeet_version());

    /* Create transcriber */
    parakeet_transcriber_t t =
        parakeet_transcriber_create(weights, vocab, NULL);
    if (!t) {
        fprintf(stderr, "Error creating transcriber: %s\n",
                parakeet_last_error());
        return 1;
    }

    /* Basic transcription */
    parakeet_result_t r =
        parakeet_transcriber_transcribe_file(t, audio_path, NULL);
    if (!r) {
        fprintf(stderr, "Error transcribing: %s\n", parakeet_last_error());
        parakeet_transcriber_free(t);
        return 1;
    }

    printf("Text: %s\n\n", parakeet_result_text(r));
    parakeet_result_free(r);

    /* Transcribe with timestamps */
    parakeet_options_t opts = parakeet_options_create();
    parakeet_options_set_timestamps(opts, 1);

    r = parakeet_transcriber_transcribe_file(t, audio_path, opts);
    if (r) {
        printf("Word timestamps:\n");
        size_t n = parakeet_result_word_count(r);
        for (size_t i = 0; i < n; i++) {
            parakeet_word_timestamp_t w = parakeet_result_word_at(r, i);
            printf("  [%.2fs - %.2fs] (%.2f) %s\n", w.start, w.end,
                   w.confidence, w.word);
        }
        parakeet_result_free(r);
    }

    parakeet_options_free(opts);
    parakeet_transcriber_free(t);

    return 0;
}
