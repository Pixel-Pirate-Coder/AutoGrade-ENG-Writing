from typing import Union
import torch
import torch.nn as nn
from torch.optim import AdamW
from bert import BERTFinetune, BERTDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import time
from datetime import datetime

import numpy as np
import pandas as pd
from pathlib import Path

class ContentScoreRegressor:
    def __init__(self,
                model: Union[BERTFinetune],
                train_dataset: Union[BERTDataset],
                test_dataset: Union[BERTDataset],
                optimizer: Union[AdamW],
                runs_folder: str,
                batch_size: int = 16,
                epochs: int = 10,
                ):
        self.model = model
        self.train_dataloader = DataLoader(train_dataset, sampler=RandomSampler, batch_size=batch_size)
        self.test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler, batch_size=batch_size)
        self.runs_folder = Path(runs_folder)
        self.model_folder = Path(self.model.model_name)
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = nn.MSELoss()
        self.optimizer = optimizer
        print(f"Using: {self.device}")

    def train(self, evaluation=True):
        print("Start training...\n")

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
                b_labels = b_labels.type(torch.LongTensor)
                b_labels = b_labels.to(self.device)

                # Zero out any previously calculated gradients
                self.model.zero_grad()

                # Perform a forward pass. This will return logits.
                logits = self.model(b_input_ids, b_attn_mask)

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
                    batch_loss, batch_counts = 0, 0, 0, 0, 0
                    t0_batch = time.time()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(self.train_dataloader)
            train_loss_list.append(avg_train_loss)

            print("-"*150)

            # model_folder = Path(self.model.model_name)
            # current_run_folder = Path("run_" + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            # output_file =  + "_" + str(epoch_i) + ".pt"

            # print("Saving model to %s" % output_file)
            # torch.save(self.model, output_file)

            # =======================================
            #               Test
            # =======================================
            if evaluation == True:
                # After the completion of each training epoch, measure the model's performance
                # on our validation set.
                test_loss = self.evaluate()
                test_loss_list.append(test_loss)

                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch

                print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Test Loss':^10} | {'Elapsed':^9}")
                print("-"*150)
                print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {test_loss:^10.6f} | {time_elapsed:^9.2f}")
                print("-"*150)

            print("\n")

        metric_df = pd.DataFrame(
            np.array([train_loss_list, test_loss_list]).T,
            columns=["train_loss", "test_loss"])

        print(metric_df)
        metric_df.to_csv(self.runs_folder.joinpath(self.model_folder).joinpath("metrics.csv"), encoding="utf_8_sig")
        print("Training complete!")
        return metric_df

    def evaluate(self):
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        self.model.eval()

        # Tracking variables
        total_loss = 0

        # For each batch in our validation set...
        for batch in self.test_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)
            b_labels = b_labels.type(torch.LongTensor)
            b_labels = b_labels.to(self.device)

            # Compute logits
            with torch.no_grad():
                logits = self.model(b_input_ids, b_attn_mask)

            # Compute loss
            loss = self.loss_fn(logits, b_labels)
            #val_loss.append(loss.item())
            total_loss += loss.item()

        # Compute the average accuracy and loss over the validation set.
        test_loss = total_loss / len(self.test_dataloader)

        # print("QWK: ", cohen_kappa_score(full_b_labels, full_preds, weights="quadratic"))

        return test_loss
                