from datasets import load_dataset
import pandas as pd
from datasets import Dataset
import transformers

class preprocess_dataset():
    def __init__(self, tokenizer):
        """
        Initializes the preprocess_dataset class.

        Args:
            tokenizer: The tokenizer to be used for tokenizing the text data.

        This method loads a specific dataset, preprocesses it, and tokenizes it.
        The tokenized dataset is then split into training and validation sets
        to be used later in the model training process.
        """

        # Initialize the tokenizer.
        self.tokenizer = tokenizer

        # Name of the dataset to be loaded from Hugging Face.
        self.dataset_name = "MaziyarPanahi/legalkit_sharegpt"

        # Preprocess the dataset.
        dataset = self.preprocess_dataset_legalkit_sharegpt()

        # Tokenize the preprocessed dataset.
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)

        # Remove the original text and label columns after tokenization.
        tokenized_dataset = tokenized_dataset.remove_columns(["text", "label"])

        # Split the tokenized dataset into training and validation sets.
        train_val_dataset = tokenized_dataset.train_test_split(test_size=0.2)
        self.train_dataset = train_val_dataset["train"]
        self.eval_dataset = train_val_dataset["test"]

    def preprocess_dataset_legalkit_sharegpt(self):
        """
        Preprocesses the LegalKit-ShareGPT dataset.

        This function loads the dataset and extracts relevant information
        from it, specifically the conversation inputs and their corresponding
        labels. It then converts the data into a format compatible with 
        Hugging Face's Dataset class for further processing.

        Returns:
            dataset: A Hugging Face Dataset object with preprocessed data.
        """

        # Load the dataset from Hugging Face.
        data = load_dataset(self.dataset_name)

        # Convert the dataset into a pandas DataFrame for easier manipulation.
        df = pd.DataFrame.from_dict(data['train'])

        # Extract the input text (questions) and the corresponding labels (responses).
        df["text"] = df["conversations_with_input"].apply(lambda x: x[0]["value"])
        df["label"] = df["conversations_with_input"].apply(lambda x: x[1]["value"])

        # Drop the original conversation columns as they are no longer needed.
        df = df.drop(columns=["conversations_with_input", "conversations"])

        # Convert the pandas DataFrame back to a Hugging Face Dataset.
        dataset = Dataset.from_pandas(df)

        return dataset

    def tokenize_function(self, examples):
        """
        Tokenizes the input and output text data.

        Args:
            examples: A batch of examples containing the "text" (questions) and "label" (responses) fields.

        This function tokenizes both the input (questions) and output (responses) texts, 
        and combines them into a single format suitable for model training.

        Returns:
            A dictionary containing tokenized input IDs, labels, and attention masks.
        """

        # Extract the questions and responses.
        questions = examples["text"]
        responses = examples["label"]

        # Tokenize the questions.
        question_tokens = self.tokenizer(questions, padding="max_length", truncation=True, max_length=512)

        # Tokenize the responses.
        response_tokens = self.tokenizer(responses, padding="max_length", truncation=True, max_length=512)

        # Combine the tokenized inputs and labels.
        input_ids = question_tokens["input_ids"]
        labels = response_tokens["input_ids"]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": question_tokens["attention_mask"]
        }


class ModelParser:
    def __init__(self, model):
        """
        Initializes the ModelParser class with a given model.

        Args:
            model: The model (e.g., a PyTorch or Hugging Face model) that will be analyzed 
                   and modified by the methods of this class.
        """
        self.model = model

    def count_parameters(self):
        """
        Counts and displays the total number of parameters in the model, 
        along with the number of trainable parameters.
        Also calculates and displays the percentage of trainable parameters 
        relative to the total number of parameters.
        """
        # Calculate the total number of parameters in the model.
        total_params = sum(p.numel() for p in self.model.parameters())

        # Calculate the number of trainable parameters (those with requires_grad=True).
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Calculate the percentage of trainable parameters.
        trainable_percentage = (trainable_params / total_params) * 100

        # Format the numbers with thousands separators for readability.
        total_params_str = f"{total_params:,}".replace(",", " ")
        trainable_params_str = f"{trainable_params:,}".replace(",", " ")

        # Print out the results.
        print(f"Total parameters: {total_params_str}")
        print(f"Trainable parameters: {trainable_params_str} ({trainable_percentage:.2f}%)\n")

    def freeze_layers_by_param_count(self, max_trainable_params):
        """
        Freezes model layers until the specified maximum number of trainable parameters is reached.

        Args:
            max_trainable_params: The maximum desired number of trainable parameters.
        """
        current_trainable_params = 0

        # Iterate through the model's parameters.
        for param in self.model.parameters():
            # If adding this parameter would exceed the max, freeze it (set requires_grad=False).
            if current_trainable_params + param.numel() > max_trainable_params:
                param.requires_grad = False
            else:
                # Otherwise, add it to the count of trainable parameters.
                current_trainable_params += param.numel()

        # Print the final count of trainable parameters after freezing.
        print(f"Final trainable parameters: {self.count_parameters}")

    def freeze_layers_by_name(self, layer_names):
        """
        Freezes specific model layers based on the provided layer names.

        Args:
            layer_names: A list of layer names to be frozen.
        """
        # Iterate through named parameters in the model.
        for name, param in self.model.named_parameters():
            # If the parameter's name matches any of the provided layer names, freeze it.
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False

        # Print the names of the layers that were frozen.
        print(f"Layers frozen: {layer_names}")

    def freeze_all_except_layer(self, layer_name_to_keep):
        """
        Freezes all layers in the model except the specified one.

        Args:
            layer_name_to_keep: The name of the layer to keep unfrozen.
        """
        # Iterate through named parameters in the model.
        for name, param in self.model.named_parameters():
            # If the layer is not the one to keep, freeze it. Otherwise, keep it trainable.
            if layer_name_to_keep not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Print the name of the layer that was kept unfrozen.
        print(f"All layers frozen except: {layer_name_to_keep}")

    def train_specific_layers(self, layer_names):
        """
        Freezes all model layers except the specified ones to be trained.

        Args:
            layer_names: A list of layer names to keep trainable.
        """
        # Iterate through named parameters in the model.
        for name, param in self.model.named_parameters():
            # If the layer's name matches any of the specified layer names, keep it trainable.
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                print(f"Layer frozen: {name}")
            else:
                # Otherwise, freeze it.
                param.requires_grad = False

    def train_lm_head(self):
        """
        Freezes all parameters except those in the language model head (`lm_head`).
        """
        # Freeze all parameters in the model.
        for param in self.model.parameters():
            param.requires_grad = False

        # If the model has an `lm_head`, unfreeze its parameters.
        if hasattr(self.model, 'lm_head'):
            for param in self.model.lm_head.parameters():
                param.requires_grad = True
            print("lm_head has been frozen.")
        else:
            # If there is no `lm_head`, print a warning.
            print("No lm_head found in the model.")

        # Count and display the number of parameters after freezing.
        self.count_parameters()
