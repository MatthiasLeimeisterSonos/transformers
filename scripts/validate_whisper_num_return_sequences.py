import sys
import datasets
from datasets import Dataset
from datasets import Audio
from transformers import (
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
        WhisperTokenizer,
        set_seed,
    )


def load_dataset():
    return datasets.load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
        )


def load_datasamples(dataset, num_samples, seed=0):
    speech_samples = dataset.sort("id").shuffle(seed).select(range(num_samples))[:num_samples]["audio"]
    return [x["array"] for x in speech_samples]


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 123
    audio_filename = sys.argv[2] if len(sys.argv) > 2 else None

    model_id = "openai/whisper-tiny.en"

    torch_device = "cpu"
    set_seed(seed)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    tokenizer = WhisperTokenizer.from_pretrained(model_id)
    model.to(torch_device)

    if audio_filename is None:
        dataset = load_dataset()
    else:
        dataset = Dataset.from_dict(
            {"audio": [audio_filename], "id": [audio_filename]}).cast_column("audio", Audio(sampling_rate=16000))

    num_beams = 5
    num_return_sequences = 5

    input_speech = load_datasamples(dataset, 1, seed)
    input_features = feature_extractor(input_speech, return_tensors="pt", sampling_rate=16_000).input_features

    print("------------------------------------------------------------------")
    print(f"Running {model_id} with {num_beams} beams and {num_return_sequences} return sequences...")

    # Perform beam search
    output = model.generate(
        input_features,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        return_dict_in_generate=True,
        output_scores=True,
    )

    hypotheses = [tokenizer.decode(output_ids, skip_special_tokens=True) for output_ids in output.sequences]
    for i, hypothesis in enumerate(hypotheses):
        print(f"  Hypothesis {i + 1}: {hypothesis}")