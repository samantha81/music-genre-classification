import torch

def train_one_epoch(model, loader, criterion, optimiser, device, clip_grad=1.0):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        mels = batch[0].to(device)
        labels = batch[1].to(device)
        
        optimiser.zero_grad()
        logits = model(mels)
        loss = criterion(logits, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimiser.step()
        
        running_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        
    return running_loss / total, correct / total


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        mels = batch[0].to(device)
        labels = batch[1].to(device)
        logits = model(mels)
        loss = criterion(logits, labels)
        
        running_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        
    return running_loss / total, correct / total


@torch.inference_mode()
def majority_vote_accuracy(model, dataset, device, batch_size=64):
    model.eval()
    song_labels = {}
    for _, label, song_id in dataset.segments:
        song_labels[song_id] = label
        
    song_preds = {sid: [] for sid in song_labels}
    for start in range(0, len(dataset.segments), batch_size):
        chunk = dataset.segments[start : start + batch_size]
        mels = torch.stack([
            torch.load(sp, weights_only=True) for sp, _, _ in chunk
        ]).to(device)
        
        preds = model(mels).argmax(1).cpu().tolist()
        for pred, (_, _, sid) in zip(preds, chunk):
            song_preds[sid].append(pred)
            
    correct = 0
    for sid, votes in song_preds.items():
        values, counts = torch.tensor(votes).unique(return_counts=True)
        if values[counts.argmax()].item() == song_labels[sid]:
            correct += 1
            
    return correct / len(song_preds)


def train_one_epoch_seg(model, loader, criterion, optimiser, device, clip_grad=1.0):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for segments, labels, _ in loader:
        segments = segments.to(device)
        labels = labels.to(device)
        
        optimiser.zero_grad()
        logits = model(segments)
        loss = criterion(logits, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimiser.step()
        
        running_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        
    return running_loss / total, correct / total


@torch.inference_mode()
def evaluate_seg(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for segments, labels, _ in loader:
        segments = segments.to(device)
        labels = labels.to(device)
        
        logits = model(segments)
        loss = criterion(logits, labels)
        
        running_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        
    return running_loss / total, correct / total
