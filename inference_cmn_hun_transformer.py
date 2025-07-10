import torch
from pathlib import Path
from data.cmn_hun import TranslationDataset

# Working directory, cache files and models will be placed in this directory
# 这个是原本的transform模型 (教师模型 - 工作正常)
base_dir = "./train_process/transformer-cmn-hun"
work_dir = Path(base_dir)

# Trained models will be placed in this directory
model_dir = Path(base_dir + "/transformer_checkpoints")

# Define training device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training data
data_dir = "data/hun-eng/hun.txt"

dataset = TranslationDataset(data_dir)

# Maximum sentence length - using teacher model configuration
max_seq_length = 32  # Teacher model supports longer sequences


def translate_english_to_hungarian(src: str):
    """
    :param src: English sentence, e.g. "I like machine learning."
    :return: Translated sentence, e.g. "Szeretem a gépi tanulást"
    """
    model = torch.load(model_dir / 'best.pt')
    model.to(device)
    model = model.eval()

    # Get model's actual max sequence length
    model_max_len = model.positional_encoding.pe.size(1)
    safe_max_len = min(max_seq_length, model_max_len - 1)

    # Tokenize and limit input length
    src_tokens = dataset.zh_vocab(dataset.zh_tokenizer(src))
    if len(src_tokens) > safe_max_len - 3:  # Reserve space for <bos>, <eos>, and generation
        src_tokens = src_tokens[:safe_max_len - 3]

    src_tensor = torch.tensor([0] + src_tokens + [1]).unsqueeze(0).to(device)

    # First tgt is <bos>
    tgt = torch.tensor([[0]]).to(device)

    # Predict word by word until predicting <eos> or reaching maximum sentence length
    for _ in range(safe_max_len):
        if tgt.size(1) >= safe_max_len:
            break

        # Perform transformer calculation
        out = model(src_tensor, tgt)
        # Prediction result, since we only need to look at the last word, take out[:, -1]
        predict = model.predictor(out[:, -1])
        # Find the index of the maximum value
        y = torch.argmax(predict, dim=1)
        # Concatenate with previous prediction results
        tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)
        # If it's <eos>, prediction ends, break the loop
        if y == 1:
            break

    # Concatenate predicted tokens
    token_list = tgt.squeeze().tolist()
    # Filter out special tokens and out-of-vocabulary tokens
    filtered_tokens = [token for token in token_list if token < len(dataset.hun_vocab) and token not in [0, 1, 2]]

    if filtered_tokens:
        result = " ".join(dataset.hun_vocab.lookup_tokens(filtered_tokens))
    else:
        result = "[translation failed]"

    return result


def translate_hungarian_to_english(src: str):
    """
    :param src: Hungarian sentence, e.g. "Szeretem a gépi tanulást"
    :return: Translated sentence, e.g. "I like machine learning."
    """
    model = torch.load(model_dir / 'best.pt')
    model.to(device)
    model = model.eval()

    # Get model's actual max sequence length
    model_max_len = model.positional_encoding.pe.size(1)
    safe_max_len = min(max_seq_length, model_max_len - 1)

    # Tokenize and limit input length
    src_tokens = dataset.hun_vocab(dataset.hun_tokenizer(src))
    if len(src_tokens) > safe_max_len - 3:  # Reserve space for <bos>, <eos>, and generation
        src_tokens = src_tokens[:safe_max_len - 3]

    src_tensor = torch.tensor([0] + src_tokens + [1]).unsqueeze(0).to(device)

    # First tgt is <bos>
    tgt = torch.tensor([[0]]).to(device)

    # Predict word by word until predicting <eos> or reaching maximum sentence length
    for _ in range(safe_max_len):
        if tgt.size(1) >= safe_max_len:
            break

        # Perform transformer calculation
        out = model(src_tensor, tgt)
        # Prediction result, since we only need to look at the last word, take out[:, -1]
        predict = model.predictor(out[:, -1])
        # Find the index of the maximum value
        y = torch.argmax(predict, dim=1)
        # Concatenate with previous prediction results
        tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)
        # If it's <eos>, prediction ends, break the loop
        if y == 1:
            break

    # Concatenate predicted tokens
    token_list = tgt.squeeze().tolist()
    # Filter out special tokens and out-of-vocabulary tokens
    filtered_tokens = [token for token in token_list if token < len(dataset.zh_vocab) and token not in [0, 1, 2]]

    if filtered_tokens:
        result = " ".join(dataset.zh_vocab.lookup_tokens(filtered_tokens))
    else:
        result = "[translation failed]"

    return result


def main():
    # Check if model file exists
    if not Path(model_dir / 'best.pt').exists():
        print(f"Model file does not exist: {model_dir / 'best.pt'}")
        print("Please train the model first or check if the path is correct")
        return

    print("=== English to Hungarian Translator ===")
    print("Type 'quit' to exit.")
    print("Just enter an English sentence:")

    while True:
        user_input = input("\nEnter English text: ").strip()

        if user_input.lower() == 'quit':
            break

        if user_input:
            print(f"English: {user_input}")
            try:
                hungarian_result = translate_english_to_hungarian(user_input)
                print(f"Hungarian: {hungarian_result}")
            except Exception as e:
                print(f"Translation error: {e}")
        else:
            print("Please enter a valid English sentence.")



# Test examples
if __name__ == "__main__":
    print(translate_hungarian_to_english("Jó reggelt!"))

    main()
