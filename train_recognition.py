import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets_loader.recursive_recognition_dataset import RecursiveRecognitionDataset
from custom_models.recognition.crnn_model import CRNN
from custom_utils.config import DEVICE, RECOGNITION_NUM_CLASSES, DATA_DIR, BATCH_SIZE, LEARNING_RATE, EPOCHS_STEPS
from custom_utils.evaluation import compute_CER
from tqdm import tqdm

def train_recognition(model, dataloader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(dataloader, desc=f"Epoch {epoch}"):
        imgs = imgs.to(config.DEVICE)
        # Forward pass (output shape: [T, B, nclass])
        preds = model(imgs)
        # NOTE: In a full implementation, convert labels (strings) to target sequences.
        # Here we create dummy targets for demonstration.
        target = torch.randint(1, config.RECOGNITION_NUM_CLASSES, (imgs.size(0), 10)).to(config.DEVICE)
        pred_lengths = torch.full((imgs.size(0),), preds.size(0), dtype=torch.long)
        target_lengths = torch.full((imgs.size(0),), 10, dtype=torch.long)
        preds_log_softmax = nn.functional.log_softmax(preds, 2)
        loss = criterion(preds_log_softmax, target, pred_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate_recognition(model, dataloader):
    model.eval()
    total_cer = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(config.DEVICE)
            preds = model(imgs)
            # Dummy decoding placeholder: replace with actual decoding method.
            pred_texts = ["sample" for _ in range(imgs.size(0))]
            for pred, gt in zip(pred_texts, labels):
                total_cer += evaluation.compute_CER(pred, gt)
    avg_cer = total_cer / len(dataloader)
    print(f"Validation CER: {avg_cer:.4f}")
    return avg_cer

def main():
    train_dataset = RecursiveRecognitionDataset(os.path.join(config.DATA_DIR, 'train'))
    val_dataset = RecursiveRecognitionDataset(os.path.join(config.DATA_DIR, 'val'))
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    # Initialize CRNN: image height=32, RGB, nclass from config, hidden units=256 (example)
    model = CRNN(imgH=32, nc=3, nclass=config.RECOGNITION_NUM_CLASSES, nh=256).to(config.DEVICE)
    criterion = nn.CTCLoss(blank=0)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    current_epoch = 0
    for stage, epochs in enumerate(config.EPOCHS_STEPS):
        print(f"\nStarting Recognition Stage {stage+1}: {epochs} epochs")
        for epoch in range(1, epochs + 1):
            current_epoch += 1
            train_recognition(model, train_loader, criterion, optimizer, current_epoch)
        print("Evaluating Recognition on Validation Set...")
        evaluate_recognition(model, val_loader)
        torch.save(model.state_dict(), f"crnn_stage{stage+1}.pt")

if __name__ == "__main__":
    main()
