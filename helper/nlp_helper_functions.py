import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from helper import *
import torch
import torch.nn as nn
import torch.optim as optim

import random
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    Timer,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from sklearn.metrics import (
    recall_score,
    accuracy_score,
)
from IPython.display import display as dp
from helper import *
from typing import Dict, List, Literal, Any, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
)
from torchmetrics.classification import BinaryAccuracy, BinaryRecall

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)


class FakeNewsDataset(Dataset):
    """
    A custom PyTorch Dataset class designed for the Fake News detection task. It processes a pandas DataFrame containing text data and corresponding labels.

    Attributes:
        df (pandas.DataFrame): The input DataFrame containing `text` and `label` columns.
        index (numpy.ndarray): An array of DataFrame indices.
        label (numpy.ndarray): An array of labels corresponding to each text entry.
        text (numpy.ndarray): An array of text entries.

    Methods:
        __len__: Returns the length of the dataset.
        __getitem__: Retrieves a single data sample, including index, text, and label.
    """

    def __init__(self, df):
        super().__init__()
        self.df = df
        self.index = self.df.index.to_numpy()
        self.label = self.df.label.to_numpy()
        self.text = self.df.text.to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        index = self.index[idx]
        text = self.text[idx]
        labels = torch.tensor(self.label[idx], dtype=torch.float32)

        return {"index": index, "text": text, "labels": labels}


class FakeNewsDataLoader(pl.LightningDataModule):
    """
        A PyTorch Lightning DataModule for efficiently loading and tokenizing the Fake News dataset.

    Attributes:
        dataset_dict (dict): Dictionary containing training, validation, testing, and optional demo datasets.
        batch_size (int): Batch size for data loaders.
        tokenizer (AutoTokenizer): Hugging Face tokenizer initialized with `distilbert-base-cased`.

    Methods:
        setup: Prepares datasets for different training stages (fit, test, demo).
        worker_init_fn: Ensures reproducibility by setting seeds for data loader workers.
        _collate_fn: Handles tokenization and prepares batches for input into the model.
        train_dataloader: Returns the training DataLoader.
        val_dataloader: Returns the validation DataLoader.
        test_dataloader: Returns the testing DataLoader.
        demo_dataloader: Returns the demo DataLoader.
    """

    def __init__(
        self,
        dataset_dict,
        batch_size=16,
    ):
        super().__init__()
        self.dataset_dict = dataset_dict
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = self.dataset_dict["train"]
            self.val_dataset = self.dataset_dict["val"]
        elif stage == "test":
            self.test_dataset = self.dataset_dict["test"]
        elif stage == "demo":
            self.demo_dataset = self.dataset_dict["demo"]
        else:  # stage == None
            self.train_dataset = self.dataset_dict["train"]

    def worker_init_fn(self, worker_id):
        seed = 0
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    def _collate_fn(self, batch):
        # Collect texts and labels
        texts = [item["text"] for item in batch]
        labels = torch.stack([item["labels"] for item in batch])

        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=14,
            persistent_workers=True,
            worker_init_fn=self.worker_init_fn,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=14,
            persistent_workers=True,
            worker_init_fn=self.worker_init_fn,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=14,
            persistent_workers=True,
            worker_init_fn=self.worker_init_fn,
            collate_fn=self._collate_fn,
        )

    def demo_dataloader(self):
        return DataLoader(
            self.demo_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=14,
            persistent_workers=True,
            worker_init_fn=self.worker_init_fn,
            collate_fn=self._collate_fn,
        )


class FakeNewsModel(pl.LightningModule):
    def __init__(
        self,
        lr=0.0001,
        fine_tune_mode=False,
        pos_weight=torch.tensor(1.2729, dtype=torch.float32),
    ):
        super().__init__()
        self.lr = lr
        self.fine_tune_mode = fine_tune_mode
        self.pos_weight = pos_weight
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-cased", num_labels=1
        )

        """
        A PyTorch Lightning Module for fine-tuning a DistilBERT-based model on the Fake News detection task.

        Attributes:
            lr (float): Learning rate.
            fine_tune_mode (bool): Indicates if the entire model or only classifier layers should be trained.
            pos_weight (torch.Tensor): Weight for positive class in BCE loss.
            model (DistilBertForSequenceClassification): Pretrained DistilBERT model with a single output neuron for binary classification.
            acc_score (BinaryAccuracy): Binary accuracy metric.
            recall_score (BinaryRecall): Binary recall metric.

        Methods:
            forward: Defines forward pass for the model.
            training_step: Computes BCE loss and logs training loss.
            validation_step: Evaluates BCE loss, updates metrics, and logs validation loss.
            on_validation_epoch_end: Computes and logs metrics at the end of the validation epoch.
            predict_step: Performs inference and returns probabilities.
            configure_optimizers: Sets up AdamW optimizer and CyclicLR scheduler.
            _set_fine_tune_mode: Configures which layers are trainable.
            change_learning_rate: Dynamically updates the learning rate.
            toggle_fine_tune_mode: Toggles fine-tuning mode and updates the learning rate.
        """

        # metrics
        self.acc_score = BinaryAccuracy()
        self.recall_score = BinaryRecall()

        # set finetune mode
        self._set_fine_tune_mode()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"].float().view(-1, 1)
        logits = self.forward(input_ids, attention_mask)
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(logits, labels)
        print("labels", labels[:5])
        print("logits", logits[:5])
        self.log(name="train_loss", value=loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"].float().view(-1, 1)
        logits = self.forward(input_ids, attention_mask)
        probs = torch.sigmoid(logits)

        y_pred = (probs > 0.5).float()
        y_true = labels

        loss = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(logits, y_true)

        self.log(name="val_loss", value=loss, on_epoch=True)

        # update metrics
        self.acc_score.update(y_pred, y_true)
        self.recall_score.update(y_pred, y_true)

        return loss

    def on_validation_epoch_end(self, outputs=None) -> None:
        self.log("acc_score", self.acc_score.compute(), on_epoch=True)
        self.log("recall_score", self.recall_score.compute(), on_epoch=True)
        self.acc_score.reset()
        self.recall_score.reset()

    def predict_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        logits = self.forward(input_ids, attention_mask).squeeze()
        probs = torch.sigmoid(logits)

        return probs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr / 40, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.lr / 40,
            max_lr=self.lr,
            step_size_up=38,
            mode="triangular",
            cycle_momentum=False,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _set_fine_tune_mode(self):
        if self.fine_tune_mode:  # fine_tune_mode: True
            for name, param in self.model.named_parameters():
                param.requires_grad = True
            print("Fine_tune_mode: True")
        else:  # fine_tune_mode: False
            for name, param in self.model.named_parameters():
                if "pre_classifier" in name or "classifier" in name:
                    param.requires_grad = True
            print("Fine_tune_mode: False")

    def change_learning_rate(self, new_lr):
        self.lr = new_lr
        print(f"Update learning rate to {self.lr}")

    def toggle_fine_tune_mode(self, updated_fine_tune_mode, new_lr):
        self.fine_tune_mode = updated_fine_tune_mode
        self._set_fine_tune_mode()
        self.lr = new_lr
        print(f"Change learning rate to:{self.lr}")


class TimeLoggingCallback(pl.Callback):
    """
    A PyTorch Lightning callback that logs the training duration.

    Attributes:
        train_start_time (float): The start time of the training.
    """

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Records the start time of the training.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning Trainer instance.
            pl_module (pl.LightningModule): The PyTorch Lightning model instance.
        """
        self.train_start_time = time.time()

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Calculates and logs the training duration when training ends.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning Trainer instance.
            pl_module (pl.LightningModule): The PyTorch Lightning model instance.
        """
        train_end_time = time.time()
        train_duration = train_end_time - self.train_start_time
        if trainer.logger is not None:
            trainer.logger.log_metrics({"train_duration": train_duration})


def train_model(
    model: pl.LightningModule,
    dataloader: pl.LightningDataModule,
    model_name: str,
    epochs: int,
    model_checkpoint: pl.Callback,
    timer: pl.Callback,
) -> None:
    """
    Args:
        model (pl.LightningModule): The PyTorch Lightning model to be trained.
        dataloader (pl.LightningDataModule): The PyTorch Lightning DataModule containing train/validation data.
        model_name (str): The name of the model used for logging purposes.
        epochs (int): The number of training epochs.
        model_checkpoint (pl.Callback): A callback for saving the model checkpoints during training.
        timer (pl.Callback): A callback for logging training time.

    Returns:
        None

    Description:
        This function sets up and trains a PyTorch Lightning model with callbacks for early stopping, learning rate monitoring, time logging, and model checkpoints. It uses CSV and TensorBoard loggers for tracking training progress. The model is trained using a GPU with mixed-precision training for improved efficiency.

        Callbacks:
            - EarlyStopping: Stops training if validation loss doesn't improve after a certain number of epochs.
            - LearningRateMonitor: Logs the learning rate at each training step.
            - TimeLoggingCallback: Logs the training time.
            - ModelCheckpoint: Saves the model's state at specified intervals.
            - Timer: Logs the elapsed time for training.

        Loggers:
            - CSVLogger: Logs training data in CSV format.
            - TensorBoardLogger: Logs training data for visualization in TensorBoard.
    """

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    time_logging = TimeLoggingCallback()

    callbacks = [early_stopping, lr_monitor, model_checkpoint, time_logging, timer]

    csv_logger = CSVLogger(save_dir="logs/thu/", name=model_name)
    tb_logger = TensorBoardLogger("tb_logs/thu/", name=model_name)

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        accelerator="gpu",
        logger=[csv_logger, tb_logger],
        precision=16,
        gradient_clip_val=1.0,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    trainer.fit(model, dataloader)


def train_head_pipeline(
    dataloader: pl.LightningDataModule,
    model: pl.LightningModule,
    model_name: str,
    epoch: int,
) -> pl.LightningModule:
    """
    train_head_pipeline
    --------------------
    Trains the head of a model using the specified data and configurations. This function performs model training for a specified number of epochs, monitors validation loss, and logs the training time.

    Args:
        dataloader (pl.LightningDataModule): The PyTorch Lightning DataModule containing train/validation data.
        model (pl.LightningModule): The PyTorch Lightning model to be trained, with its head to be trained specifically.
        model_name (str): The name of the model, used for logging purposes.
        epoch (int): The number of epochs for training.

    Returns:
        pl.LightningModule: The trained model.

    Description:
        This function trains the head of the model (typically the final layer or classification head) while utilizing a model checkpoint to save the best model based on validation loss. It also tracks and logs the time taken for training.

        Callbacks:
            - ModelCheckpoint: Saves the best model based on validation loss.
            - Timer: Logs the training time.

        The training time is printed in a formatted "minutes:seconds" format once training is complete.

        The `train_model` function is called to handle the actual training.
    """
    
    head_train_model_cp = ModelCheckpoint(
        save_top_k=1,
        mode="min",
        monitor="val_loss",
        filename="{epoch:02d}-{val_loss:.2f}",
    )

    timer = Timer()
    train_model(
        model=model,
        dataloader=dataloader,
        model_name=model_name,
        epochs=epoch,
        model_checkpoint=head_train_model_cp,
        timer=timer,
    )
    formatted_time = time.strftime("%M:%S", time.gmtime(timer.time_elapsed("train")))
    print(f"{model_name} head training time ({epoch} epochs):")
    print(formatted_time)

    return model



def fine_tune_pipeline(
    dataloader: pl.LightningDataModule,
    head_trained_model: pl.LightningModule,
    model_name: str,
    batch_size: int,
    lr: float,
    epoch: int,
) -> pl.LightningModule:
    """
    Fine-tunes a pre-trained model with specified data loader, saves the best model checkpoint based on validation loss,
    and prints the training time.

    Args:
        dataloader (pl.LightningDataModule): The PyTorch Lightning DataModule containing the data.
        head_trained_model (pl.LightningModule): The pre-trained PyTorch Lightning model to be fine-tuned.
        model_name (str): The name of the model for logging purposes.
        batch_size (int): The batch size for training.
        lr (float): The learning rate for fine-tuning.
        epoch (int): The number of epochs for fine-tuning.

    Returns:
        pl.LightningModule: The fine-tuned PyTorch Lightning model.
    """
    model = head_trained_model
    model.toggle_fine_tune_mode(True, lr)

    model_cp = ModelCheckpoint(
        save_top_k=1,
        mode="min",
        monitor="val_loss",
        filename="{epoch:02d}-{val_loss:.2f}",
    )

    timer = Timer()
    train_model(
        model=model,
        dataloader=dataloader,
        model_name=model_name,
        epochs=epoch,
        model_checkpoint=model_cp,
        timer=timer,
    )

    formatted_time = time.strftime("%M:%S", time.gmtime(timer.time_elapsed("train")))
    print(f"{model_name} head training time ({epoch} epochs):")
    print(formatted_time)

    return model


def prediction_pipeline(
    dataloader: pl.LightningDataModule,
    fine_tuned_model: pl.LightningModule,
    dataset_type: str = "test",
) -> List[Any]:
    """
    Runs a prediction pipeline on a given test dataset using a fine-tuned model.

    Args:
        dataloader (pl.LightningDataModule): The data module containing the test dataset.
        fine_tuned_model (pl.LightningModule): The fine-tuned model for making predictions.

    Returns:
        List[Any]: A list of predictions generated by the model on the test dataset.
    """
    dataloader.setup("test")
    model = fine_tuned_model
    if dataset_type == "val":
        test_dataloader = dataloader.val_dataloader()
    else:
        test_dataloader = dataloader.test_dataloader()
    trainer = pl.Trainer(enable_model_summary=False, enable_progress_bar=False)

    predictions = trainer.predict(model, dataloaders=test_dataloader)

    return predictions


def initial_lr_finder(model: nn.Module, data_loader: Any, batch_size: int):
    """
    initial_lr_finder
    -----------------
    This function finds the optimal learning rate for training by using a cyclical learning rate scheduler. It iterates through a demo data loader to collect loss and learning rate values, helping to visualize the learning rate range and identify the best starting learning rate for training.

    Args:
        model (nn.Module): The neural network model to train, typically a PyTorch model.
        data_loader (Any): The data loader that provides the training data. It is assumed to have a `demo_dataloader` method for demo purposes.
        batch_size (int): The batch size to be used during training.

    Returns:
        pd.DataFrame: A DataFrame containing two columns: "lr" (learning rate) and "loss" (corresponding loss at each step).

    Description:
        The function uses a cyclical learning rate scheduler to vary the learning rate between a `base_lr` and `max_lr` while training the model on a demo dataset. The loss at each learning rate is recorded and returned in a DataFrame for analysis. The goal is to help identify an appropriate learning rate for the actual training phase based on the learning rate vs. loss curve.

        The function handles interruptions (e.g., keyboard interrupts) gracefully and ensures that data collected up to the interruption is returned.
    """
    
    pl.seed_everything(0)
    set_seed(0)

    data_loader.setup(stage="demo")
    demo_data_loader = data_loader.demo_dataloader()
    dataset_len = len(data_loader.demo_dataloader().dataset)

    # set optimizer
    optimizer = optim.SGD(model.parameters())
    # set cyclical learning rate scheduler
    step_size_up = int(np.ceil(dataset_len / batch_size))

    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1e-5,
        max_lr=5e-3,
        step_size_up=step_size_up,
        mode="triangular",
    )
    criterion = nn.BCEWithLogitsLoss()

    lr_ls = []
    loss_ls = []

    try:
        # loop through batches to collect lr and loss
        for batch_id, batch in enumerate(demo_data_loader):
            logits = model.forward(batch["input_ids"], batch["attention_mask"])
            y_true = batch["labels"].float().view(-1, 1)
            loss = criterion(logits, y_true)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # update gradient
            optimizer.step()

            # update the scheduler
            scheduler.step()

            # get lr and loss
            current_lr = optimizer.param_groups[0]["lr"]
            lr_ls.append(current_lr)
            loss_ls.append(loss.item())
            print("loss:", loss.item())

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving collected data...")

        if len(lr_ls) != len(loss_ls):
            print("Lists length mismatch, return a dictionary.")
            return {"lr_ls": lr_ls, "loss_ls": loss_ls}

    finally:
        loss_lr_df = pd.DataFrame({"lr": lr_ls, "loss": loss_ls})

        return loss_lr_df



def detect_all_caps(text: str, threshold: float = 0.1) -> bool:
    """
    Detects if the given text contains a proportion of uppercase words exceeding the specified threshold.

    Args:
        text (str): The input text to analyze.
        threshold (float, optional): The proportion threshold for determining if the text contains a significant number of uppercase words. Defaults to 0.1.

    Returns:
        bool: True if the proportion of uppercase words in the text exceeds the threshold, False otherwise.
    """
    text = (
        text.replace("I ", "i ").replace("UTC", "utc").replace("REDIRECT", "redirect")
    )
    text_ls = text.split()
    text_len = len(text_ls)

    return sum(i.isupper() for i in text_ls) / text_len > threshold


def set_seed(seed: int) -> None:
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    This function sets the seed for Python's `random` module, NumPy, and PyTorch.
    Additionally, it configures PyTorch to use deterministic algorithms to further
    ensure reproducible results. If CUDA is available, the seed is also set for
    CUDA operations.

    Args:
        seed (int): The seed value to use for random number generation.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_pred(
    y_pred_prob: np.ndarray,
    y_true: np.ndarray,
    model_name: str,
    show_cf: bool = False,
    show_metric: bool = False,
) -> None:
    """
    Evaluates the model predictions against the true labels using various metrics and visualizations.

    Args:
        y_pred_prob (np.ndarray): An array of prediction probabilities.
        y_true (np.ndarray): An array of true labels.
        model_name (str): The name of the model for reporting purposes.
        pred_thres (Optional[List[float]], default=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]): List of thresholds for converting probabilities to binary predictions.
        show_pr_curve (bool, default=False): Whether to plot precision-recall curves.
        show_cf (bool, default=False): Whether to plot confusion matrices.
        show_metric (bool, default=False): Whether to display a dataframe of metrics.

    Returns:
        None: This function does not return a value. It produces plots and prints results.
    """

    y_pred = [1 if i > 0.5 else 0 for i in y_pred_prob]

    labels = ["real", "fake"]

    recall = recall_score(y_true, y_pred, average="weighted")
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")

    metrics_df = pd.DataFrame(
        {"accuracy": accuracy, "recall": recall, "roc_auc": roc_auc},
        index=labels,
    )

    if show_cf:
        cf = confusion_matrix(y_true, y_pred)
        fig_size(3, 3)
        sns.heatmap(cf, annot=True, fmt=".0f", cbar=False, cmap="coolwarm")

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.xticks([0.5, 1.5], labels, ha="right")
        plt.yticks([0.5, 1.5], labels, ha="right")

        plt.title(f"{model_name} Test Data Prediction")

        plt.show()

    if show_metric:
        dp(metrics_df.style.apply(mark_df_color, id=1, axis=1, color="mistyrose"))


def lime_predictor(texts):
    """
    This function uses a fine-tuned DistilBERT model to predict the likelihood of text belonging to two classes, which can be used with LIME (Local Interpretable Model-agnostic Explanations) for model interpretability. It takes in a list of texts, tokenizes them, and passes them through a pre-trained model to obtain predictions.

    Args:
        texts (list of str): A list of text inputs to be classified by the model.

    Returns:
        np.ndarray: A 2D array of shape (len(texts), 2) containing the predicted probabilities for each class, where the first column represents the probability of the negative class and the second column represents the probability of the positive class.

    Description:
        The function loads a fine-tuned DistilBERT model, tokenizes the input texts using the corresponding tokenizer, and predicts the probability distribution of two classes for each text. The predictions are returned as a NumPy array, which can be used with model-agnostic interpretability methods like LIME.
    """
    
    distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    best_fine_tuned_path = (
        "logs/thu/distilbert-0.0001/version_1/checkpoints/epoch=01-val_loss=0.01.ckpt"
    )
    best_fine_tuned_model = FakeNewsModel.load_from_checkpoint(best_fine_tuned_path)

    # mimic a batch
    batch = distilbert_tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt",
    )
    batch["input_ids"] = batch["input_ids"].to("cuda")
    batch["attention_mask"] = batch["attention_mask"].to("cuda")

    with torch.no_grad():
        logits = best_fine_tuned_model.forward(
            batch["input_ids"], batch["attention_mask"]
        )

        pos_probs = torch.sigmoid(logits).cpu().detach().numpy()
        neg_probs = 1 - pos_probs

        torch.cuda.empty_cache()

        return np.stack([neg_probs, pos_probs], axis=1)[:, :, 0]



def plot_head_met(
    df,
    plot_title="Cased DistilBERT Head Training (bach_size=32, epoch=3)",
    only_one_epoch=False,
):
    """
    plot_head_met
    -------------
    This function generates plots to visualize the training metrics (loss, accuracy, recall) of the model head during training. It uses the provided DataFrame containing metrics data and generates either scatter or line plots depending on the number of epochs.

    Args:
        df (pandas.DataFrame): A DataFrame containing the training metrics for each epoch, including 'epoch', 'val_loss', 'train_loss_epoch', 'acc_score', and 'recall_score'.
        plot_title (str, optional): The title to be displayed on the plots. Default is "Cased DistilBERT Head Training (bach_size=32, epoch=3)".
        only_one_epoch (bool, optional): If True, scatter plots are used to visualize the metrics for one epoch; otherwise, line plots are used for multiple epochs. Default is False.

    Returns:
        None: The function will display the plots, but it does not return any value.

    Description:
        The function visualizes the model's training performance by plotting:
        1. **Train and Validation Loss**: Shows the loss values for both the training and validation sets over epochs.
        2. **Accuracy**: Displays the accuracy score over epochs.
        3. **Recall**: Displays the recall score over epochs.

        If `only_one_epoch` is set to True, scatter plots are generated for each metric. Otherwise, line plots are used to display the changes over multiple epochs.
    """
    
    print(
        f"Total train duration: {sec_to_min_sec(df.train_duration.tolist()[-1])} seconds."
    )

    if only_one_epoch:
        fig_size(14, 2)
        plt.subplot(1, 3, 1)
        sns.scatterplot(data=df, x="epoch", y="val_loss", label="val")
        sns.scatterplot(data=df, x="epoch", y="train_loss_epoch", label="train")
        plt.title("Train and Validation Loss")
        plt.subplot(1, 3, 2)
        sns.scatterplot(data=df, x="epoch", y="acc_score", label="val")
        plt.title("Accuracy")
        plt.subplot(1, 3, 3)
        sns.scatterplot(data=df, x="epoch", y="recall_score", label="val")
        plt.title("Recall")
        plt.suptitle(plot_title, y=1.2)
        plt.subplots_adjust(wspace=0.4)
        plt.show()
    else:
        fig_size(14, 2)
        plt.subplot(1, 3, 1)
        sns.lineplot(data=df, x="epoch", y="val_loss", label="val")
        sns.lineplot(data=df, x="epoch", y="train_loss_epoch", label="train")
        plt.title("Train and Validation Loss")
        plt.subplot(1, 3, 2)
        sns.lineplot(data=df, x="epoch", y="acc_score", label="val")
        plt.title("Accuracy")
        plt.subplot(1, 3, 3)
        sns.lineplot(data=df, x="epoch", y="recall_score", label="val")
        plt.title("Recall")
        plt.suptitle(plot_title, y=1.2)
        plt.subplots_adjust(wspace=0.4)
        plt.show()

