from typing import Union, Callable
import torch
import torch.nn as nn
from torch.optim import AdamW
from .bert import BERTFinetune, BERTDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import time
from datetime import datetime

import numpy as np
import pandas as pd
from pathlib import Path

class ScoreRegressor:
    def __init__(self,
                model: Union[BERTFinetune],
                train_dataset: Union[BERTDataset],
                test_dataset: Union[BERTDataset],
                optimizer: Union[AdamW],
                metric_function: Callable[[torch.tensor, torch.tensor], float],
                runs_folder: str,
                batch_size: int = 16,
                epochs: int = 10,
                scale_predictions: bool = True
                ):
        self.model = model
        self.train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
        self.test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)
        self.epochs = epochs
        self.scale_predictions = scale_predictions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = nn.MSELoss()
        self.optimizer = optimizer
        self.metric_function = metric_function
        self.runs_folder = Path(runs_folder)
        self.model_folder = Path(self.model.model_name)
        self.metrics_df = pd.DataFrame()
        
        # Get maximum value of train target
        if hasattr(train_dataset, "__max__"):
            self.max_target = train_dataset.__max__
        else:
            if self.scale_predictions:
                print("`__max__` attribute should be added to train dataset in order to scale predictions")
            self.max_target = None
        
        # Get minimum value of train target
        if hasattr(train_dataset, "__min__"):
            self.min_target = train_dataset.__min__
        else:
            if self.scale_predictions:
                print("`__min__` attribute should be added to train dataset in order to scale predictions")
            self.min_target = None
        
        print(f"Using device: {self.device}")

    def train(self, evaluation=True, save_best=True) -> None:
        print("Start training...\n")
        
        current_run_folder = self.runs_folder\
            .joinpath(self.model_folder)\
                .joinpath(Path("run_" + datetime.now().strftime('%Y-%m-%dT%H-%M-%S')))
        current_run_folder.mkdir(parents=True, exist_ok=True)

        train_loss_list = []
        test_loss_list = []

        for epoch_i in range(self.epochs):
            # =======================================
            #               Training
            # =======================================
            print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12}| {'Val Loss':^10} | {'Elapsed':^9}")
            print("-"*150)

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0

            # Put the model into the training mode
            self.model.train()

            # For each batch of training data...
            for step, batch in enumerate(self.train_dataloader):
                batch_counts += 1
                # Load batch to GPU
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)
                b_labels = b_labels.type(torch.FloatTensor)
                b_labels = b_labels.to(self.device)

                # Zero out any previously calculated gradients
                self.model.zero_grad()

                # Perform a forward pass. This will return logits.
                logits = self.scale_output(self.model(b_input_ids, b_attn_mask))

                # Compute loss and accumulate the loss values
                loss = self.loss_fn(logits, b_labels)
                batch_loss += loss.item()
                total_loss += loss.item()

                # Perform a backward pass to calculate gradients
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and the learning rate
                self.optimizer.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (step == len(self.train_dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^10} | {'-':^10} | {'-':^10} | {time_elapsed:^9.2f}")

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(self.train_dataloader)
            # Save best model weights
            if not evaluation:
                if save_best:
                    if train_loss_list:
                        if train_loss_list[-1] > avg_train_loss:
                            self.save_model_weights(current_run_folder)
                else:
                    self.save_model_weights(current_run_folder)
            train_loss_list.append(avg_train_loss)

            print("-"*150)

            # =======================================
            #               Test
            # =======================================
            if evaluation:
                # After the completion of each training epoch, measure the model's performance
                # on our validation set.
                test_loss = self.evaluate()
                
                # Save best model weights
                if save_best:
                    if test_loss_list:
                        if test_loss_list[-1] > test_loss:
                            self.save_model_weights()
                    else:
                        self.save_model_weights(current_run_folder)
                        
                test_loss_list.append(test_loss)

                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch

                print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Test Loss':^10} | {'Elapsed':^9}")
                print("-"*150)
                print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {test_loss:^10.6f} | {time_elapsed:^9.2f}")
                print("-"*150)

            print("\n")
        self.metric_df = pd.DataFrame(
            np.array([train_loss_list, test_loss_list]).T,
            columns=["train_loss", "test_loss"])

        self.metric_df.to_csv(self.runs_folder.joinpath(self.model_folder).joinpath("metrics.csv"), encoding="utf_8_sig")
        print("Training complete!")
        self.save_model_weights(current_run_folder, name="last") # Save last model weights

    def evaluate(self) -> float:
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        self.model.eval()

        # Tracking variables
        total_loss = 0
        full_b_labels, full_preds = [], []

        # For each batch in our validation set...
        for batch in self.test_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)
            b_labels = b_labels.type(torch.FloatTensor)
            b_labels = b_labels.to(self.device)

            # Compute logits
            with torch.no_grad():
                logits = self.scale_output(self.model(b_input_ids, b_attn_mask))

            # Compute loss
            loss = self.loss_fn(logits, b_labels)
            total_loss += loss.item()
            
            full_b_labels.extend(b_labels.cpu().data)
            full_preds.extend(logits.cpu().data)

        # Compute the average accuracy and loss over the validation set.
        test_loss = total_loss / len(self.test_dataloader)

        print("Evaluation Metric: ", self.metric_function(full_b_labels, full_preds))

        return test_loss
    
    def scale_output(self, predictions: torch.tensor) -> torch.tensor:
        """Scales model predictions to the same scale as train targets

        Parameters
        ----------
        predictions : torch.tensor
            Model predictions in [0, 1] scale

        Returns
        -------
        torch.tensor
            Scaled model predictions in [min_target, max_target] scale
        """    
        if self.scale_predictions and self.max_target is not None and self.min_target is not None:
            return self.min_target + (self.max_target - self.min_target) * predictions
        else:
            return predictions
    
    def save_model_weights(self, save_folder: Path, name: str = "best") -> None:
        """Saves model weigths to a .pt file in current run folder

        Parameters
        ----------
        name : str, optional
            Name of model .pt file, by default "best"
        """
        output_file = save_folder.joinpath("%s.pt" % name)
        print("Saving {} model to {}".format(name, output_file))
        torch.save(self.model, output_file)
                