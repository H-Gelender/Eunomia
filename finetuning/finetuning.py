%%writefile model_finetuning.py
from datasets import load_dataset, Dataset
import pandas as pd
import transformers
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, get_linear_schedule_with_warmup
import torch
from torch.optim import Adam

from typing import Dict, Optional, Sequence


class PreprocessDatasetLegalkitShareGPT:
    """
    A class to preprocess the 'MaziyarPanahi/legalkit_sharegpt' dataset for training a model.

    Attributes
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        A tokenizer from the Hugging Face Transformers library.
    dataset_name : str
        The name of the dataset to be loaded and processed.
    train_dataset : datasets.Dataset
        The tokenized and split training dataset.
    eval_dataset : datasets.Dataset
        The tokenized and split validation dataset.
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        """
        Initialize the PreprocessDatasetLegalkitShareGPT class with a tokenizer.
        
        Parameters
        ----------
        tokenizer : transformers.PreTrainedTokenizer
            The tokenizer used for processing text data.
        """
        self.tokenizer = tokenizer
        self.dataset_name = "MaziyarPanahi/legalkit_sharegpt"
        
        # Preprocess the dataset and tokenize it
        dataset = self.preprocess_dataset_legalkit_sharegpt()
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(["text", "label"])

        # Split the tokenized dataset into training and validation sets
        train_val_dataset = tokenized_dataset.train_test_split(test_size=0.2)
        self.train_dataset = train_val_dataset["train"]
        self.eval_dataset = train_val_dataset["test"]

    def preprocess_dataset_legalkit_sharegpt(self) -> Dataset:
        """
        Load and preprocess the 'MaziyarPanahi/legalkit_sharegpt' dataset.
        
        Returns
        -------
        Dataset
            A Hugging Face Dataset object containing the processed text and label columns.
        """
        # Load the dataset
        data = load_dataset(self.dataset_name)
        
        # Convert the dataset to a pandas DataFrame for easier manipulation
        df = pd.DataFrame.from_dict(data['train'])

        # Extract the text and label columns from the conversation data
        df["text"] = df["conversations_with_input"].apply(lambda x: x[0]["value"])
        df["label"] = df["conversations_with_input"].apply(lambda x: x[1]["value"])

        # Drop unnecessary columns
        df = df.drop(columns=["conversations_with_input", "conversations"])

        # Convert the pandas DataFrame back to a Hugging Face Dataset
        dataset = Dataset.from_pandas(df)

        return dataset

    def tokenize_function(self, examples: dict) -> dict:
        """
        Tokenize the input text and labels.

        Parameters
        ----------
        examples : dict
            A dictionary containing the examples to be tokenized.

        Returns
        -------
        dict
            A dictionary with tokenized input_ids, labels, and attention_mask.
        """
        questions = examples["text"]
        responses = examples["label"]

        # Tokenize the questions
        question_tokens = self.tokenizer(
            questions, padding="max_length", truncation=True, max_length=512
        )

        # Tokenize the responses
        response_tokens = self.tokenizer(
            responses, padding="max_length", truncation=True, max_length=512
        )

        # Combine the tokenized inputs and labels
        input_ids = question_tokens["input_ids"]
        labels = response_tokens["input_ids"]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": question_tokens["attention_mask"],
        }

class ModelParser:
    """
    A class to manage and modify the trainability of layers in a machine learning model.
    
    Attributes
    ----------
    model : torch.nn.Module or transformers.PreTrainedModel
        The model whose parameters will be analyzed and modified.
    """

    def __init__(self, model):
        """
        Initialize the class with the given model.

        Parameters
        ----------
        model : torch.nn.Module or transformers.PreTrainedModel
            The model to be processed (e.g., a PyTorch model or a Hugging Face model).
        """
        self.model = model

    def count_parameters(self):
        """
        Count and display the total number of parameters in the model, 
        along with the number of trainable parameters.
        Also, display the percentage of trainable parameters relative to the total number.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        trainable_percentage = (trainable_params / total_params) * 100

        # Format the numbers with thousands separators
        total_params_str = f"{total_params:,}".replace(",", " ")
        trainable_params_str = f"{trainable_params:,}".replace(",", " ")

        print(f"Total parameters: {total_params_str}")
        print(f"Trainable parameters: {trainable_params_str} ({trainable_percentage:.2f}%)\n")

    def freeze_layers_by_param_count(self, max_trainable_params: int):
        """
        Freeze layers in the model until the maximum number of trainable parameters is reached.

        Parameters
        ----------
        max_trainable_params : int
            The desired maximum number of trainable parameters.
        """
        current_trainable_params = 0

        # Iterate through the model's layers
        for param in self.model.parameters():
            if current_trainable_params + param.numel() > max_trainable_params:
                param.requires_grad = False
            else:
                current_trainable_params += param.numel()

        print(f"Final trainable parameters: {self.count_parameters()}")

    def freeze_layers_by_name(self, layer_names: list):
        """
        Freeze the layers of the model based on the provided layer names.

        Parameters
        ----------
        layer_names : list
            A list of layer names to freeze.
        """
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False

       

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict[str, str],
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """
    Resize the tokenizer and the corresponding model embeddings to account for new special tokens.

    This function adds special tokens to the tokenizer, resizes the model's token embeddings,
    and initializes the embeddings for the new tokens by averaging the existing token embeddings.
    
    Note: This method may result in the embedding size not being divisible by 64.

    Parameters
    ----------
    special_tokens_dict : Dict[str, str]
        A dictionary of special tokens to be added to the tokenizer.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer to which the special tokens will be added.
    model : transformers.PreTrainedModel
        The model whose token embeddings will be resized.

    Returns
    -------
    model : transformers.PreTrainedModel
        The model with resized token embeddings.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer with added special tokens.
    """
    # Add the special tokens to the tokenizer and resize the model's embeddings
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        # Get the input and output embeddings from the model
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        # Calculate the average of the existing embeddings
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        # Assign the averaged embeddings to the new tokens
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    return model, tokenizer
    
def init_llama(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Initialize the model and tokenizer with default special tokens if not already set.
    
    Args:
        model (PreTrainedModel): The model to initialize.
        tokenizer (PreTrainedTokenizer): The tokenizer to initialize.
        
    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: The updated model and tokenizer.
    """
    # Default special tokens
    DEFAULT_PAD_TOKEN: str = "[PAD]"
    DEFAULT_EOS_TOKEN: str = "</s>"
    DEFAULT_BOS_TOKEN: str = "<s>"
    DEFAULT_UNK_TOKEN: str = "<unk>"

    # Dictionary to store special tokens
    special_tokens_dict: dict = {}

    # Add special tokens to the dictionary if they are not already set
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # Resize the tokenizer and model embeddings based on the special tokens
    model, tokenizer = smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    return model, tokenizer

class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
        """
        Create the optimizer and scheduler for training.

        Args:
            num_training_steps (int): The total number of training steps.
        """
        if self.optimizer is None:
            self.optimizer = optimizer  # Use the optimizer defined in the script
        if self.lr_scheduler is None:
            self.lr_scheduler = scheduler  # Use the scheduler defined in the script

if __name__ == '__main__':
    # Define the model and tokenizer
    model_name: str = "meta-llama/Meta-Llama-3.1-8B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Preprocess the dataset
    prep_data = preprocess_dataset_legalkit_sharegpt(tokenizer)
    model, tokenizer = init_llama(model, tokenizer)
    train_data = prep_data.train_dataset
    val_data = prep_data.eval_dataset
    data_collator = transformers.data.data_collator.default_data_collator

    # Define training parameters
    num_train_steps: int = 300
    optimizer = Adam(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./Model_llama_3_1_8B",
        fp16=False,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=2,
        learning_rate=1e-4,
        evaluation_strategy='no',
        save_strategy='no',
        max_steps=num_train_steps,
        logging_steps=5,
    )

    # Initialize and train the CustomTrainer
    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
