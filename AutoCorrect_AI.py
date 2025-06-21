import tkinter as tk
from tkinter import ttk, messagebox
import nltk
import re
import random
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Download Brown Corpus
nltk.download('brown')
from nltk.corpus import brown

class AutocorrectApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hybrid Autocorrect System")
        self.root.geometry("800x600")
        
        # Initialize the autocorrect system
        self.initialize_autocorrect()
        
        # Create GUI components
        self.create_widgets()
        
    def initialize_autocorrect(self):
        # Load and preprocess text data
        self.words = brown.words()
        self.clean_words = [w.lower() for w in self.words if w.isalpha()]

        # Count word frequencies
        self.word_freq = Counter(self.clean_words)
        self.vocab = set(self.word_freq.keys())

        # Character tokenizer
        self.char_vocab = sorted(set("abcdefghijklmnopqrstuvwxyz")) + ['<PAD>', '<SOS>', '<EOS>']
        self.char2idx = {ch: i for i, ch in enumerate(self.char_vocab)}
        self.idx2char = {i: ch for i, ch in enumerate(self.char_vocab)}
        self.vocab_size = len(self.char_vocab)

        # Create training data
        self.top_words = [w for w, _ in self.word_freq.most_common(5000)]
        self.training_data = [(self.make_typo(w), w) for w in self.top_words for _ in range(2) 
                            if (typo := self.make_typo(w)) != w and typo not in self.word_freq]

        # Training Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder(self.vocab_size, 128, 256).to(self.device)
        self.decoder = Decoder(self.vocab_size, 128, 256).to(self.device)
        self.model = Seq2Seq(self.encoder, self.decoder, self.device).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.char2idx['<PAD>'])

        # Training Loop
        self.dataset = TypoDataset(self.training_data)
        self.loader = DataLoader(self.dataset, batch_size=64, shuffle=True)

        # Train the model in a background thread to keep the GUI responsive
        import threading
        training_thread = threading.Thread(target=self.train_model)
        training_thread.start()

    def train_model(self):
        for epoch in range(10):
            self.model.train()
            total_loss = 0
            for src, trg in self.loader:
                src, trg = src.to(self.device), trg.to(self.device)
                output = self.model(src, trg, teacher_forcing_ratio=0.5)
                output = output[:, 1:].reshape(-1, self.vocab_size)
                trg = trg[:, 1:].reshape(-1)
                loss = self.criterion(output, trg)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/10 | Loss: {total_loss/len(self.loader):.4f}")
        
        # Update status when training is complete
        self.status_label.config(text="Model training complete!", fg="green")

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Hybrid Autocorrect System", font=('Helvetica', 16, 'bold'))
        title_label.pack(pady=10)

        # Input section
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=10)

        ttk.Label(input_frame, text="Enter word to correct:").pack(side=tk.LEFT)
        self.word_entry = ttk.Entry(input_frame, width=30)
        self.word_entry.pack(side=tk.LEFT, padx=10)
        self.word_entry.bind("<Return>", lambda e: self.correct_word())

        correct_button = ttk.Button(input_frame, text="Correct", command=self.correct_word)
        correct_button.pack(side=tk.LEFT)

        # Results section
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        ttk.Label(results_frame, text="Correction Results:").pack(anchor=tk.W)
        self.results_text = tk.Text(results_frame, height=10, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # Test words section
        test_frame = ttk.Frame(main_frame)
        test_frame.pack(fill=tk.X, pady=10)

        ttk.Label(test_frame, text="Test with sample words:").pack(anchor=tk.W)
        test_button_frame = ttk.Frame(test_frame)
        test_button_frame.pack(fill=tk.X, pady=5)

        test_words = ["speling", "exmple", "autocorrect", "mistake", "writting"]
        for word in test_words:
            btn = ttk.Button(test_button_frame, text=word, 
                           command=lambda w=word: self.test_word(w))
            btn.pack(side=tk.LEFT, padx=5)

        # Status bar
        self.status_label = ttk.Label(main_frame, text="Initializing model...", relief=tk.SUNKEN)
        self.status_label.pack(fill=tk.X, pady=10)

    def correct_word(self):
        word = self.word_entry.get().strip().lower()
        if not word:
            messagebox.showwarning("Input Error", "Please enter a word to correct")
            return
        
        if not word.isalpha():
            messagebox.showwarning("Input Error", "Please enter alphabetic characters only")
            return
        
        corrected = self.correct_typo(word)
        
        self.results_text.insert(tk.END, f"Input: {word}\n")
        self.results_text.insert(tk.END, f"Corrected: {corrected}\n")
        self.results_text.insert(tk.END, "-"*50 + "\n")
        self.results_text.see(tk.END)
        self.word_entry.delete(0, tk.END)

    def test_word(self, word):
        self.word_entry.delete(0, tk.END)
        self.word_entry.insert(0, word)
        self.correct_word()

    # Rule-based autocorrect functions
    def edits1(self, word):
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def known(self, words): 
        return set(w for w in words if w in self.vocab)
    
    def candidates(self, word): 
        return self.known([word]) or self.known(self.edits1(word)) or [word]
    
    def autocorrect(self, word): 
        return max(self.candidates(word), key=lambda w: self.word_freq[w])

    # Enhanced typo generation
    def make_typo(self, word):
        if len(word) < 4: return word
        operations = random.randint(1, 2) if len(word) > 6 else 1
        typo = word
        for _ in range(operations):
            i = random.randint(1, len(typo)-2)
            op = random.choice(['swap', 'drop', 'replace', 'double'])
            if op == 'swap' and i < len(typo)-1:
                typo = typo[:i] + typo[i+1] + typo[i] + typo[i+2:]
            elif op == 'drop':
                typo = typo[:i] + typo[i+1:]
            elif op == 'replace':
                typo = typo[:i] + random.choice('abcdefghijklmnopqrstuvwxyz') + typo[i+1:]
            elif op == 'double':
                typo = typo[:i] + typo[i] + typo[i:]
        return typo

    def encode(self, word, max_len=15):
        encoded = [self.char2idx['<SOS>']] + [self.char2idx.get(c, 0) for c in word.lower() if c in self.char2idx]
        encoded = encoded[:max_len-1] + [self.char2idx['<EOS>']]
        return encoded + [self.char2idx['<PAD>']] * (max_len - len(encoded))

    def decode(self, indices):
        return ''.join([self.idx2char[i] for i in indices if self.idx2char[i] not in ['<PAD>', '<SOS>', '<EOS>']])

    # Hybrid Correction Function
    def correct_typo(self, word):
        # First try neural net
        self.model.eval()
        with torch.no_grad():
            encoded = torch.tensor([self.encode(word)]).to(self.device)
            hidden, cell = self.model.encoder(encoded)
            
            input_token = torch.tensor([self.char2idx['<SOS>']]).to(self.device)
            decoded = []
            
            for _ in range(len(word) + 5):
                output, hidden, cell = self.model.decoder(input_token, hidden, cell)
                pred_token = output.argmax().item()
                if pred_token == self.char2idx['<EOS>']:
                    break
                decoded.append(pred_token)
                input_token = torch.tensor([pred_token]).to(self.device)
            
            neural_suggestion = self.decode(decoded)
        
        # Validate neural suggestion
        if (len(neural_suggestion) >= 3 and 
            neural_suggestion in self.vocab and 
            abs(len(neural_suggestion) - len(word)) <= 2):
            return neural_suggestion
        
        # Fallback to rule-based
        return self.autocorrect(word)

# Model Architecture
class Encoder(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_size, emb_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(output_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        embedded = self.embedding(x.unsqueeze(1))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        return self.fc(output.squeeze(1)), hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        outputs = torch.zeros(batch_size, trg_len, self.decoder.fc.out_features).to(self.device)
        
        hidden, cell = self.encoder(src)
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output
            input = trg[:, t] if random.random() < teacher_forcing_ratio else output.argmax(1)
            
        return outputs

# Dataset
class TypoDataset(Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        typo, correct = self.pairs[idx]
        return torch.tensor(encode(typo)), torch.tensor(encode(correct))

def encode(word, max_len=15):
    char_vocab = sorted(set("abcdefghijklmnopqrstuvwxyz")) + ['<PAD>', '<SOS>', '<EOS>']
    char2idx = {ch: i for i, ch in enumerate(char_vocab)}
    encoded = [char2idx['<SOS>']] + [char2idx.get(c, 0) for c in word.lower() if c in char2idx]
    encoded = encoded[:max_len-1] + [char2idx['<EOS>']]
    return encoded + [char2idx['<PAD>']] * (max_len - len(encoded))

if __name__ == "__main__":
    root = tk.Tk()
    app = AutocorrectApp(root)
    root.mainloop()