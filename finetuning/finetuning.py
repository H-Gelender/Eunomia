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

class preprocess_dataset_legalkit_sharegpt():
    def __init__(self, tokenizer):

        self.tokenizer = tokenizer
        self.dataset_name = "MaziyarPanahi/legalkit_sharegpt"
        self.dataset = self.preprocess_dataset_legalkit_sharegpt()
        tokenized_dataset = self.dataset.map(self.tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(["text", "label"])

        # Structurer pour le Trainer
        train_val_dataset = tokenized_dataset.train_test_split(test_size=0.2)
        self.train_dataset = train_val_dataset["train"]
        self.eval_dataset = train_val_dataset["test"]

    def preprocess_dataset_legalkit_sharegpt(self):

        data = load_dataset(self.dataset_name)
        df = pd.DataFrame.from_dict(data['train'])

        df["text"] = df["conversations_with_input"].apply(lambda x: x[0]["value"])
        df["label"] = df["conversations_with_input"].apply(lambda x: x[1]["value"])

        df = df.drop(columns = ["conversations_with_input"])
        df = df.drop(columns = ["conversations"])

        dataset = Dataset.from_pandas(df)

        return dataset

    def tokenize_function(self, examples):

        questions = examples["text"]
        responses = examples["label"]

        # Tokeniser les questions
        question_tokens = self.tokenizer(questions, padding="max_length", truncation=True, max_length=512)

        # Tokeniser les réponses
        response_tokens = self.tokenizer(responses, padding="max_length", truncation=True, max_length=512)

        # Les combiner en une seule entrée
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
        Initialise la classe avec un modèle donné.

        Args:
            model: Le modèle (par exemple, un modèle PyTorch ou Hugging Face).
        """
        self.model = model

    def count_parameters(self):
        """
        Compte et affiche le nombre total de paramètres du modèle ainsi que le nombre de paramètres entraînables.
        Affiche également le pourcentage de paramètres entraînables par rapport au nombre total de paramètres.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        trainable_percentage = (trainable_params / total_params) * 100

            # Formater les nombres avec des séparateurs de milliers
        total_params_str = f"{total_params:,}".replace(",", " ")
        trainable_params_str = f"{trainable_params:,}".replace(",", " ")

        print(f"Total parameters: {total_params_str}")
        print(f"Trainable parameters: {trainable_params_str} ({trainable_percentage:.2f}%)\n")


    def freeze_layers_by_param_count(self, max_trainable_params):
        """
        Gèle les couches du modèle jusqu'à ce que le nombre maximum de paramètres entraînables soit atteint.

        Args:
            max_trainable_params: Le nombre maximum de paramètres entraînables souhaité.
        """
        current_trainable_params = 0

        # Parcourir les couches du modèle
        for param in self.model.parameters():
            if current_trainable_params + param.numel() > max_trainable_params:
                param.requires_grad = False
            else:
                current_trainable_params += param.numel()

        print(f"Final trainable parameters: {self.count_parameters}")

    def freeze_layers_by_name(self, layer_names):
        """
        Gèle les couches du modèle en fonction des noms de couches fournis.

        Args:
            layer_names: Une liste de noms de couches à geler.
        """
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False

        print(f"Layers frozen: {layer_names}")

    def freeze_all_except_layer(self, layer_name_to_keep):
        """
        Gèle toutes les couches du modèle sauf celle spécifiée par son nom.

        Args:
            layer_name_to_keep: Le nom de la couche à conserver non gelée.
        """
        for name, param in self.model.named_parameters():
            if layer_name_to_keep not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        print(f"All layers frozen except: {layer_name_to_keep}")

    def train_specific_layers(self, layer_names):
        """
        Gèle les couches du modèle en fonction des noms de couches fournis.

        Args:
            layer_names: Une liste de noms de couches à geler.
        """
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                print(f"Layer frozen: {name}")
            else:
                param.requires_grad = False

    def train_lm_head(self):
        """
        Gèle tous les paramètres sauf la couche lm_head.
        """
        for param in self.model.parameters():
            param.requires_grad = False

        if hasattr(self.model, 'lm_head'):
            for param in self.model.lm_head.parameters():
                param.requires_grad = True
            print("lm_head has been frozen.")
        else:
            print("No lm_head found in the model.")

        self.count_parameters()

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    ):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
    return model, tokenizer
    
def init_llama(model, tokenizer):
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"


    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    model, tokenizer = smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    return model, tokenizer
    
class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps):
        if self.optimizer is None:
            self.optimizer = optimizer
        if self.lr_scheduler is None:
            self.lr_scheduler = scheduler
            
if __name__ == 'main':

    model_name = "meta-llama/Meta-Llama-3.1-8B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prep_data = preprocess_dataset_legalkit_sharegpt(tokenizer)
    model, tokenizer = init_llama(model, tokenizer)
    train_data = prep_data.train_dataset
    val_data = prep_data.eval_dataset
    data_collator = transformers.data.data_collator.default_data_collator

    num_train_steps = 300
    optimizer = Adam(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)



    training_args = TrainingArguments(
        "./Model_llama_3_1_8B",
        fp16=False,
        gradient_accumulation_steps= 1,
        per_device_train_batch_size = 2,
        learning_rate = 1e-4,
        evaluation_strategy = 'no',
        save_strategy = 'no',
        max_steps=num_train_steps,
        logging_steps=5,

    )

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        args=training_args,
        data_collator=data_collator,
        tokenizer = tokenizer,
    )

    trainer.train()
