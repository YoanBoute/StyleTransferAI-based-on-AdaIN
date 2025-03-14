import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import mlflow
from pathlib import Path
import json
import pickle

from adain_model import AdaINModel
from adain_datamodule import AdaINDataModule

class AdaINTrainer() :
    def __init__(self, model : AdaINModel, data_module : AdaINDataModule, optimizer, mlflow_dir : Path = Path('./'), mlflow_tags : dict = {}) : 
        self.model = model
        self.dm = data_module
        self.optimizer = optimizer
        self.mlflow_dir = mlflow_dir
        self.mlflow_tags = mlflow_tags

        self._prepared = False

    def _prepare(self) :
        print("Initiating datasets...")
        try :
            self.dm.create_datasets()
        except Exception as e :
            raise Exception(f"Could not create the datasets ({str(e)})") from None
        print("Compiling the model...")
        try :
            self.model.compile(optimizer=self.optimizer)
        except Exception as e :
            raise Exception(f"Could not compile the model ({str(e)})") from None
        
        mlflow.set_tracking_uri(self.mlflow_dir)
        mlflow.set_experiment("AdaIN Style Transfer")
        self._prepared = True
        print("Model is ready to be trained")
    
    def train(self, epochs = 1, steps_per_epoch = None, log_every_n_epochs = None, validation_steps = 10, test_steps = 100) :
        if not self._prepared :
            self._prepare()
        if log_every_n_epochs is None :
            log_every_n_epochs = epochs
        torch.random.manual_seed(self.dm.seed) # Seed is fixed here so as to have reproducible training (especially to handle the RandomCrop transform and so to always have the same image pairs)
        np.random.seed(self.dm.seed)
        
        train_dataloader = self.dm.train_dataloader()
        val_dataloader = self.dm.val_dataloader()
        test_dataloader = self.dm.test_dataloader()

        num_train_loops = int(np.ceil(epochs / log_every_n_epochs))
        print("Training the model...")
        for loop in range(num_train_loops) :
            initial_epoch = loop * log_every_n_epochs 
            with mlflow.start_run() as run :
                mlflow.keras.autolog(log_models=False, silent=True)
                mlflow.set_tags(self.mlflow_tags)
                mlflow.set_tag("Initial epoch", initial_epoch)
                params = {'style_loss_weight' : self.model.loss_.lamb,
                          'loss_reduction' : self.model.loss_.reduction}
                mlflow.log_params(params)
                self.model.fit(train_dataloader, validation_data=val_dataloader, epochs=initial_epoch + min(log_every_n_epochs, epochs-initial_epoch), initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
                test_loss = self.model.evaluate(test_dataloader, steps=test_steps)
                mlflow.log_metric("test_loss", test_loss)
                self._log_model(run)
            if num_train_loops > 1 :
                print("Checkpoint reached, model has been logged")
                        
        print("The training is complete ! The model and its performances have been logged in MLFlow")
    
    def _log_model(self, run) :
        tmp_path = Path('./_tmp_load')
        if tmp_path.exists() :
            shutil.rmtree(tmp_path)
        tmp_path.mkdir()

        # Save model architecture
        with open(tmp_path / "architecture.json", "w") as f :
            f.write(self.model.to_json())
        # Save model weights
        self.model.save_weights(tmp_path / "weights.weights.h5")
       
        # Log the files as artifacts
        mlflow.log_artifacts(tmp_path, run_id=run.info.run_id)

        # Remove the temp dir after use
        shutil.rmtree(tmp_path)
