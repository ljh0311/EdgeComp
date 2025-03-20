"""
Speech Emotion Recognition Training Script
---------------------------------------
This script sets up and trains a speech emotion recognition model using SpeechBrain.
It handles:
- Environment setup and dependency installation
- Dataset preparation and preprocessing
- Model training across different hyperparameter combinations
- Checkpoint management and logging
"""

import sys
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import random
import itertools
from datetime import datetime
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb

# Install required packages
os.system("pip install speechbrain -q")
os.system("git clone https://github.com/Ishant1/SpeechAnalytics.git")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create directories
os.makedirs("/kaggle/working/sample_data", exist_ok=True)

# Load hyperparameter config
hyperparam_file = (
    "/kaggle/input/hyperparam-wav2vec2/train_with_wav2vec2_with_dropout.yml"
)

# Dataset paths and replacements
DATASET_PATHS = {
    "asvp": {
        "file": "/kaggle/input/speech-address/asvp_dict.json",
        "replace": [
            (
                "/content/gdrive/MyDrive/dataset/ERC/ASVP",
                "/kaggle/input/asvpesdspeech-nonspeech-emotional-utterances/ASVP-ESD-Update",
            )
        ],
    },
    "meld_dev": {
        "file": "/kaggle/input/speech-address/meld_dev_dict.json",
        "replace": [
            (
                "/content/gdrive/MyDrive/dataset/ERC/MELD-RAW-MP3",
                "/kaggle/input/meld-dataset/MELD-RAW",
            ),
            ("mp3", "mp4"),
        ],
    },
    "meld_train": {
        "file": "/kaggle/input/speech-address/meld_train_dict.json",
        "replace": [
            (
                "/content/gdrive/MyDrive/dataset/ERC/MELD-RAW-MP3",
                "/kaggle/input/meld-dataset/MELD-RAW",
            ),
            ("mp3", "mp4"),
        ],
    },
    "cremad": {
        "file": "/kaggle/input/speech-address/cremad_dict.json",
        "replace": [
            ("/content/gdrive/MyDrive/dataset/ERC/CREMAD", "/kaggle/input/cremad")
        ],
    },
    "iemocap": {
        "file": "/kaggle/input/speech-address/iemocap_dict.json",
        "replace": [
            (
                "/content/gdrive/MyDrive/dataset/ERC/IEMOCAP",
                "/kaggle/input/iemocapfullrelease/IEMOCAP_full_release",
            )
        ],
    },
}

# Process datasets
replacement_dict = {k: v for k, v in DATASET_PATHS.items() if os.path.exists(v["file"])}

for dataset_name, dataset_info in replacement_dict.items():
    with open(dataset_info["file"], "r") as f:
        data_dict = json.load(f)

    for entry in data_dict.values():
        for old_path, new_path in dataset_info["replace"]:
            entry["wav"] = entry["wav"].replace(old_path, new_path)

    new_dir = "/kaggle/working/speech-address"
    os.makedirs(new_dir, exist_ok=True)

    output_file = dataset_info["file"].replace("/kaggle/input/speech-address", new_dir)
    with open(output_file, "w") as f:
        json.dump(data_dict, f)


class EmoIdBrain(sb.Brain):
    """Brain class for emotion recognition model training and evaluation"""

    def compute_forward(self, batch, stage):
        """Forward pass computation"""
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        outputs = self.modules.wav2vec2(wavs, lens)
        outputs = self.hparams.avg_pool(outputs, lens)
        outputs = outputs.view(outputs.shape[0], -1)
        outputs = self.hparams.dropout(outputs)
        outputs = self.modules.output_mlp(outputs)

        return self.hparams.log_softmax(outputs)

    def compute_objectives(self, predictions, batch, stage):
        """Compute loss"""
        emoid = batch.emo_encoded.squeeze(1)
        loss = self.hparams.compute_cost(predictions, emoid)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, emoid)

        return loss

    def fit_batch(self, batch):
        """Train one batch"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        loss.backward()
        if self.check_gradients(loss):
            self.wav2vec2_optimizer.step()
            self.optimizer.step()

        self.zero_grad()

        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        """Setup for stage start"""
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Handle stage end"""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            return

        stats = {
            "loss": stage_loss,
            "error_rate": self.error_metrics.summarize("average"),
        }

        if stage == sb.Stage.VALID:
            # Update learning rates
            old_lr, new_lr = self.hparams.lr_annealing(stats["error_rate"])
            old_lr_wav2vec2, new_lr_wav2vec2 = self.hparams.lr_annealing_wav2vec2(
                stats["error_rate"]
            )

            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec2_optimizer, new_lr_wav2vec2
            )

            # Log stats
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr, "wave2vec_lr": old_lr_wav2vec2},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save checkpoint
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error_rate"])

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current}, test_stats=stats
            )

    def init_optimizers(self):
        """Initialize optimizers"""
        self.wav2vec2_optimizer = self.hparams.wav2vec2_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("wav2vec2_opt", self.wav2vec2_optimizer)
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

    def zero_grad(self, set_to_none=False):
        """Zero out gradients"""
        self.wav2vec2_optimizer.zero_grad(set_to_none)
        self.optimizer.zero_grad(set_to_none)


def get_data_sample(dataset_dicts, save=True, data_root="sample_data"):
    """Create train/test/validation splits from datasets"""
    full_data = {}

    # Load and sample data
    for name, info in dataset_dicts["datasets"].items():
        with open(info["json"]) as f:
            data_dict = json.load(f)

        sample_size = int(info["ratio"] * len(data_dict))
        sampled_keys = random.sample(list(data_dict.keys()), sample_size)

        for key in sampled_keys:
            full_data[f"{name}_{key}"] = data_dict[key]

    # Split data
    keys = list(full_data.keys())
    labels = [d["emo"] for d in full_data.values()]

    train_val_keys, test_keys, train_val_label, test_label = train_test_split(
        keys, labels, stratify=labels, test_size=dataset_dicts["splits"]["test"]
    )

    train_val_ratio = 1 - dataset_dicts["splits"]["test"]
    val_ratio = dataset_dicts["splits"]["valid"] / train_val_ratio

    train_keys, val_keys, train_label, val_label = train_test_split(
        train_val_keys, train_val_label, stratify=train_val_label, test_size=val_ratio
    )

    # Create split datasets
    splits = {
        "train": {k: v for k, v in full_data.items() if k in train_keys},
        "test": {k: v for k, v in full_data.items() if k in test_keys},
        "val": {k: v for k, v in full_data.items() if k in val_keys},
    }

    if save:
        os.makedirs("sample_data", exist_ok=True)

        for split_name, split_data in splits.items():
            with open(f"sample_data/{split_name}.json", "w") as f:
                json.dump(split_data, f)

        keys = {
            "train": list(splits["train"].keys()),
            "test": list(splits["test"].keys()),
            "val": list(splits["val"].keys()),
        }

        with open("sample_data/keys.json", "w") as f:
            json.dump(keys, f)

    return splits["train"], splits["test"], splits["val"]


def get_data_from_ids(ids, dataset_dicts, filename=None):
    """Get data for specific IDs from datasets"""
    all_data = {}
    for name, info in dataset_dicts["datasets"].items():
        with open(info["json"]) as f:
            all_data[name] = json.load(f)

    final_data = {}
    for id_ in ids:
        dataset, uid = id_.split("_", 1)
        final_data[id_] = all_data[dataset][uid]

    if filename:
        with open(filename, "w") as f:
            json.dump(final_data, f)
        return filename

    return final_data


# Data pipeline functions
@sb.utils.data_pipeline.takes("wav")
@sb.utils.data_pipeline.provides("sig")
def audio_pipeline(wav):
    """Load audio signal"""
    return sb.dataio.dataio.read_audio(wav)


# Initialize label encoder
label_encoder = sb.dataio.encoder.CategoricalEncoder()


@sb.utils.data_pipeline.takes("emo")
@sb.utils.data_pipeline.provides("emo", "emo_encoded")
def label_pipeline(emo):
    """Process emotion labels"""
    yield emo
    yield label_encoder.encode_label_torch(emo)


# Training configuration
train_test_valid_splits = {"train": 0.5, "test": 0.3, "valid": 0.2}

dataset_dicts = {
    "datasets": {
        "cremad": {
            "json": "/kaggle/working/speech-address/cremad_dict.json",
            "ratio": 1,
        },
        "iemocap": {
            "json": "/kaggle/working/speech-address/iemocap_dict.json",
            "ratio": 0.5,
        },
        "asvp": {"json": "/kaggle/working/speech-address/asvp_dict.json", "ratio": 1},
    },
    "splits": train_test_valid_splits,
}

# Create datasets
data_root = "sample_data"
get_data_sample(dataset_dicts, save=True, data_root=data_root)
print(f"Created datasets with split: {train_test_valid_splits}")

# Load datasets
datasets = {}
data_info = {
    "train": f"{data_root}/train.json",
    "valid": f"{data_root}/dev.json",
    "test": f"{data_root}/test.json",
}

for dataset in data_info:
    datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=data_info[dataset],
        replacements={"data_root": data_root},
        dynamic_items=[audio_pipeline, label_pipeline],
        output_keys=["id", "sig", "emo_encoded"],
    )

# Load/create label encoder
lab_enc_file = os.path.join("sample_data", "label_encoder.txt")
label_encoder.load_or_create(
    path=lab_enc_file, from_didatasets=[datasets["train"]], output_key="emo"
)

# Training loop
epochs = [3, 5]
dropouts = [0.1, 0.2, 0.5]
learning_rates = [0.0001, 0.001]

print(f"Starting training at {datetime.now()}")

for epoch, dropout, lr in itertools.product(epochs, dropouts, learning_rates):
    print(f"Training: epochs={epoch}, dropout={dropout}, lr={lr}")

    output_folder = f"results/{epoch}-{dropout}-{lr}-model"
    os.environ["OUTPUT_DIR"] = output_folder

    # Configure hyperparameters
    overrides = {
        "data_folder": "/kaggle/input/iemocapfullrelease/IEMOCAP_full_release",
        "number_of_epochs": epoch,
        "dropout_prob": dropout,
        "lr": lr,
        "output_folder": output_folder,
        "wav2vec2_folder": "wav2vec2_checkpoint",
    }

    with open(hyperparam_file) as f:
        hparams = load_hyperpyyaml(f, overrides)

    hparams["wav2vec2"] = hparams["wav2vec2"].to(device)

    # Initialize and train model
    brain = EmoIdBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts={"device": device},
        checkpointer=hparams["checkpointer"],
    )

    brain.fit(
        epoch_counter=brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Save checkpoints
    save_contents = os.listdir(f"{output_folder}/save")
    checkpoint_dir = next(c for c in save_contents if "CKPT" in c)

    os.environ["CKPT_DIR"] = checkpoint_dir
    os.environ["NEW_CKPT_DIR"] = "ckpt"

    # Manage checkpoint files
    os.system(f"rm {output_folder}/save/{checkpoint_dir}/wav2vec2_opt.ckpt")
    os.makedirs(f"{output_folder}/save/ckpt", exist_ok=True)
    os.system(
        f"cp -r {output_folder}/save/{checkpoint_dir}/. {output_folder}/save/ckpt"
    )
    os.system(f"rm -r {output_folder}/save/{checkpoint_dir}")

    # Copy additional files
    for file in ["train_log.txt", "keys.json", "test.json", "label_encoder.txt"]:
        src = (
            "sample_data/" + file
            if file != "train_log.txt"
            else f"{output_folder}/" + file
        )
        os.system(f"cp {src} {output_folder}/save/ckpt")

    print(f"Finished training: epochs={epoch}, dropout={dropout}, lr={lr}")

# Cleanup
os.system("rm -r wav2vec2_checkpoint")
