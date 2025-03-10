#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A nowadays (year 2025) common contrastive loss trains a model to distinguish between similar + dissimilar speech segments:
- contrast different speakers saying same words/segments
- contrast same speaker saying different words/segments

In the context of deepfake detection and speaker similarity, a triplet loss is awesome:
Anchor (A): Base data point.
Positive (P): Data point similar to anchor.
Negative (N): Data point dissimilar to anchor.
L(A, P, N) = max(0, d(A, P) - d(A, N) + margin)

In this script I utilize data augmented speech signals as anchor (A),
in relation to their original recording before augmentation (P),
to distinct versus impostors approaches using TTS or VC (N).
"""

from transformers import AutoProcessor, AutoModelForAudioClassification, WavLMProcessor, WavLMForAudioClassification, TrainingArguments, Trainer
from datasets import load_dataset, Audio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import os

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class TripletTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        anchors = inputs.pop("anchors")
        positives = inputs.pop("positives")
        negatives = inputs.pop("negatives")

        anchor_embeddings = model(**anchors).logits
        positive_embeddings = model(**positives).logits
        negative_embeddings = model(**negatives).logits

        loss_fct = TripletLoss()
        loss = loss_fct(anchor_embeddings, positive_embeddings, negative_embeddings)
        return (loss, None) if return_outputs else loss

def prepare_triplets(dataset):
    triplets = []
    positive_dict = {}
    negative_dict = {}

    for i, example in enumerate(dataset):
        label = example['label']
        if label not in positive_dict:
            positive_dict[label] = []
        positive_dict[label].append(i)

    labels = list(positive_dict.keys())

    for anchor_idx, anchor_example in enumerate(dataset):
        anchor_label = anchor_example['label']
        if len(positive_dict[anchor_label]) < 2:
            continue #skip if there are not enough positives.

        positive_idx = np.random.choice([idx for idx in positive_dict[anchor_label] if idx != anchor_idx])
        negative_label = np.random.choice([l for l in labels if l != anchor_label])
        negative_idx = np.random.choice(positive_dict[negative_label])

        triplets.append({
            'anchors': dataset[anchor_idx],
            'positives': dataset[positive_idx],
            'negatives': dataset[negative_idx],
            'labels': anchor_label
        })

    return triplets

def fine_tune_audio_classifier_triplet(base_model_name, train_dataset_path, output_dir, num_epochs=3, batch_size=16):
    try:
        processor = WavLMProcessor.from_pretrained(base_model_name)
        model = WavLMForAudioClassification.from_pretrained(base_model_name, num_labels=2)		

        dataset = load_dataset("audiofolder", data_dir=train_dataset_path)
        dataset = dataset["train"].cast_column("audio", Audio(sampling_rate=16000))

        triplet_dataset = prepare_triplets(dataset)

        def preprocess_function(examples):
            processed_anchors = processor([x["anchors"]["audio"]["array"] for x in examples], sampling_rate=16000, return_tensors="pt", padding=True, truncation=True)
            processed_positives = processor([x["positives"]["audio"]["array"] for x in examples], sampling_rate=16000, return_tensors="pt", padding=True, truncation=True)
            processed_negatives = processor([x["negatives"]["audio"]["array"] for x in examples], sampling_rate=16000, return_tensors="pt", padding=True, truncation=True)
            labels = torch.tensor([x["labels"] for x in examples])
            return {"anchors": processed_anchors, "positives": processed_positives, "negatives": processed_negatives, "labels": labels}

        processed_dataset = dataset.map(lambda examples: {"label": examples["label"]}, batched=True).select(range(len(triplet_dataset))) #add labels to the dataset.
        processed_dataset = processed_dataset.map(lambda example, index: triplet_dataset[index], with_indices=True) #add triplet data.
        processed_dataset = processed_dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="no",
        )

        trainer = TripletTrainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset,
            tokenizer=processor,
        )

        trainer.train()

        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)

        print(f"Fine-tuned model saved to {output_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")

base_model = "microsoft/wavlm-base-plus"
output_model_path = "./fine_tuned_triplet_model"
fine_tune_audio_classifier_triplet(base_model, train_data_path, output_model_path)