// Speaker diarization with Sortformer
//
// Usage: example-diarize <sortformer.safetensors> <audio.wav>

#include <parakeet/parakeet.hpp>

#include <iomanip>
#include <iostream>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <sortformer.safetensors> <audio.wav>\n";
        return 1;
    }

    auto cfg = parakeet::make_sortformer_117m_config();
    parakeet::Sortformer model(cfg);
    model.load_state_dict(axiom::io::safetensors::load(argv[1]));

    auto audio = parakeet::read_audio(argv[2]);
    auto features =
        parakeet::preprocess_audio(audio.samples, {.normalize = false});
    auto segments = model.diarize(features);

    std::cout << "Detected " << segments.size() << " segment(s):\n\n";
    for (const auto &seg : segments) {
        std::cout << "  Speaker " << seg.speaker_id << ": [" << std::fixed
                  << std::setprecision(2) << seg.start << "s - " << seg.end
                  << "s]\n";
    }

    return 0;
}
