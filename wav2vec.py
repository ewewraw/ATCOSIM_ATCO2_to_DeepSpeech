from huggingsound import TrainingArguments, ModelArguments, SpeechRecognitionModel, TokenSet
import csv


def fine_tune():
    model = SpeechRecognitionModel("facebook/wav2vec2-large-xlsr-53")

    input_dir = "data_ready_for_training/"
    audio_data_directory_name = input_dir + "en/clips/"
    dataset_tsv_name = input_dir + "en/train.tsv"
    dataset_tsv = open(dataset_tsv_name, mode='r', encoding='utf-8')
    output_dir = "data_ready_for_training/fine_tuned"

    # first of all, you need to define your model's token set
    # however, the token set is only needed for non-finetuned models
    # if you pass a new token set for an already finetuned model, it'll be ignored during training
    tokens = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
    token_set = TokenSet(tokens)

    # define your custom train data
    train_data = []
    reader = csv.reader(dataset_tsv, delimiter="\t")
    for rownum, line in enumerate(reader):
        print(line)
        if line[1] != "path":
            try:
                train_data.append({"path": audio_data_directory_name + line[1] + ".wav", "transcription": line[2]})
            except FileNotFoundError:
                print("not found {}".format(line[1]))
        else:
            continue

    # and finally, fine-tune your model
    model.finetune(
        output_dir,
        train_data=train_data,
        token_set=token_set,
    )