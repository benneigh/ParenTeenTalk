import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Check if MPS is available, otherwise fallback to CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print( "Using MPS backend on macOS.")
else:
    device = torch.device("cpu")
    print(" MPS not available, using CPU instead.")
torch.mps.set_per_process_memory_fraction(0.0)
# Load datasets
attributes_df = pd.read_csv("Separate CSV for Attributes - All.csv")
dialogue_df = pd.read_csv("Main Dialogue Dataset - All.csv")

# Extract conversation-level ID
dialogue_df["Conversation_ID"] = dialogue_df["ID"].apply(lambda x: '-'.join(x.split('-')[:3]))
attributes_df["Conversation_ID"] = attributes_df["ID"].apply(lambda x: '-'.join(x.split('-')[:3]))

# Merge engagement scores
attributes_df["Variant_ID"] = attributes_df["Conversation_ID"].apply(lambda x: '-'.join(x.split('-')[:3]))
dialogue_df["Variant_ID"] = dialogue_df["Conversation_ID"].apply(lambda x: '-'.join(x.split('-')[:3]))
variant_scores = attributes_df[['Variant_ID', 'Engagement Score']].dropna().drop_duplicates()
merged_df = dialogue_df.merge(variant_scores, on='Variant_ID', how='left')

# Filter child utterances
child_utterances_df = merged_df[merged_df['speaker'] == 'Child']

# Drop missing engagement scores
labeled_data = child_utterances_df.dropna(subset=['Engagement Score'])
unlabeled_data = child_utterances_df[child_utterances_df['Engagement Score'].isna()]

# Convert utterances to strings
labeled_data['utterance'] = labeled_data['utterance'].astype(str)
unlabeled_data['utterance'] = unlabeled_data['utterance'].astype(str)

# Encode engagement scores as labels
label_encoder = LabelEncoder()
labeled_data['Engagement_Label'] = label_encoder.fit_transform(labeled_data['Engagement Score'])

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define dataset class
class EngagementDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts.astype(str).tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    labeled_data['utterance'], labeled_data['Engagement_Label'], test_size=0.2, random_state=42
)

# Create datasets
train_dataset = EngagementDataset(train_texts, train_labels, tokenizer)
test_dataset = EngagementDataset(test_texts, test_labels, tokenizer)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_)).to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./bert_results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./bert_logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Predict missing engagement scores
if not unlabeled_data.empty:
    unlabeled_texts = unlabeled_data['utterance'].tolist()
    encodings = tokenizer(unlabeled_texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    encodings = {key: val.to(device) for key, val in encodings.items()}  # Ensure tensors are on MPS

    model.eval()
    with torch.no_grad():
        outputs = model(**encodings)
        predictions = torch.argmax(outputs.logits, axis=1).cpu().numpy()  # Move predictions to CPU

    # Convert predictions back to engagement scores
    unlabeled_data['Engagement Score'] = label_encoder.inverse_transform(predictions)

    # Merge back into the main dataset
    merged_df.loc[unlabeled_data.index, 'Engagement Score'] = unlabeled_data['Engagement Score']

# Save updated dataset
merged_df.to_csv("Updated_Dataset_with_Engagement_Scores.csv", index=False)

print(" Fine-tuning complete. Predictions saved in 'Updated_Dataset_with_Engagement_Scores.csv'.")
