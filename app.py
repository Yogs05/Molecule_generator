import streamlit as st
import torch
import pandas as pd
import random
from collections import Counter
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO

from model.xlstm_model import xLSTMModel
from generation.sampler import generate_smiles
from chemistry.properties import compute_properties
from chemistry.filters import passes_filter




def smiles_to_image(smiles, size=(300, 300)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    img = Draw.MolToImage(mol, size=size)

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return buf


class SMILESVocabulary:
    def __init__(self, smiles_list, min_freq=5):
        self.special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
        self.build_vocabulary(smiles_list, min_freq)
        print(f"\nVocabulary size: {self.vocab_size}")
        print("Top 10 most common tokens:")
        for char, count in self.char_counter.most_common(10):
            print(f"  '{char}': {count}")

    def build_vocabulary(self, smiles_list, min_freq):
        print("Building vocabulary from cleaned SMILES...")

        # Use sample for large datasets
        if len(smiles_list) > 300000:
            sample_size = min(300000, len(smiles_list))
            smiles_list = random.sample(smiles_list, sample_size)
            print(f"Using {sample_size} sample for vocabulary")

        # Count characters
        self.char_counter = Counter()
        for smiles in tqdm(smiles_list, desc="Counting characters"):
            self.char_counter.update(list(smiles))

        # Filter by frequency AND remove non-SMILES characters
        valid_smiles_chars = set(
            'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            '0123456789'
            '@#=+/\\:.$()[]{}%-<>?&*^!~'
        )

        # Include only valid SMILES characters with minimum frequency
        filtered_chars = {char for char, count in self.char_counter.items()
                         if (count >= min_freq and char in valid_smiles_chars)}

        print(f"Original unique chars: {len(self.char_counter)}")
        print(f"After filtering: {len(filtered_chars)}")

        self.char_to_idx = {}
        self.idx_to_char = {}

        # Add special tokens
        for i, token in enumerate(self.special_tokens):
            self.char_to_idx[token] = i
            self.idx_to_char[i] = token

        # Add filtered characters
        start_idx = len(self.special_tokens)
        for i, char in enumerate(sorted(filtered_chars)):
            idx = start_idx + i
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char

        self.vocab_size = len(self.char_to_idx)
        self.pad_idx = self.char_to_idx['<PAD>']
        self.start_idx = self.char_to_idx['<START>']
        self.end_idx = self.char_to_idx['<END>']
        self.unk_idx = self.char_to_idx['<UNK>']

    def encode(self, smiles, max_length=None):
        tokens = ['<START>'] + list(smiles) + ['<END>']
        indices = [self.char_to_idx.get(token, self.unk_idx) for token in tokens]

        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length-1] + [self.end_idx]
            else:
                indices = indices + [self.pad_idx] * (max_length - len(indices))

        return indices

    def decode(self, indices):
        tokens = []
        for idx in indices:
            if idx == self.end_idx or idx == self.pad_idx:
                break
            token = self.idx_to_char.get(idx, '?')
            if token not in self.special_tokens:
                tokens.append(token)
        return ''.join(tokens)


DEVICE = "cpu"
torch.serialization.add_safe_globals([SMILESVocabulary])

# ----------------------------
# Load model + vocab (SAFE)
# ----------------------------
@st.cache_resource
def load_model():
    checkpoint = torch.load(
        "checkpoint/xlstm_generator.pth",
        map_location=DEVICE,
        weights_only = False
    )

    vocab = checkpoint["vocab"]

    model = xLSTMModel(
        vocab_size=vocab.vocab_size,
        embed_dim=256,
        hidden_dim=512,
        num_layers=2
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, vocab

model, vocab = load_model()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🧪 xLSTM Drug-like Molecule Generator")

drug_type = st.selectbox(
    "Select drug purpose",
    ["Headache", "Antibiotic", "Anti-inflammatory"]
)

num_molecules = st.slider(
    "Number of molecules",
    min_value=1,
    max_value=50,
    value=10
)

temperature = st.slider(
    "Sampling temperature",
    min_value=0.5,
    max_value=1.5,
    value=1.0,
    step=0.1
)

if st.button("Generate Molecules"):
    results = []

    with st.spinner("Generating molecules..."):
        attempts = 0
        while len(results) < num_molecules and attempts < num_molecules * 10:
            attempts += 1

            smiles = generate_smiles(
                model,
                vocab,
                temperature=temperature,
                device=DEVICE
            )

            props = compute_properties(smiles)
            if props is None:
                continue

            if passes_filter(props, drug_type):
                results.append({
                    "SMILES": smiles,
                    **props
                })

    if results:
        df = pd.DataFrame(results)

        st.success(f"Generated {len(df)} molecules")
        st.dataframe(df, use_container_width=True)

        st.subheader("🧬 Molecule Structures")

        # ✅ Create columns ONCE
        cols = st.columns(2)

        for i, row in df.iterrows():
            img = smiles_to_image(row["SMILES"])

            if img is None:
                continue

            # ✅ Safely alternate columns
            with cols[i % 2]:
                st.image(
                    img,
                    caption=row["SMILES"],
                    use_container_width=True
                )
    else:
        st.warning("No valid molecules found. Try increasing temperature.")


