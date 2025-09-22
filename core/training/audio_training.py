"""
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

"""
Audio Processing Model Training Program
Train multimodal audio processing model supporting speech recognition, tone analysis, music recognition, noise recognition, and audio synthesis
"""

import os
import json
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from tqdm import tqdm
import logging
import librosa
import soundfile as sf
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioDataset(Dataset):
    """Audio model training dataset"""
    
    def __init__(self, data_dir, feature_extractor, max_length=16000 * 5):  # 5 seconds audio
        self.data = []
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        
        # Load multimodal audio data
        audio_types = ['speech', 'music', 'noise', 'tone', 'effects']
        
        for audio_type in audio_types:
            file_path = os.path.join(data_dir, f"audio_{audio_type}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    audio_data = json.load(f)
                    self.data.extend(audio_data)
        
        logger.info(f"Loaded audio training data: {len(self.data)} samples")
    
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load audio file
        audio_path = item['audio_path']
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Truncate or pad to fixed length
            if len(audio) > self.max_length:
                audio = audio[:self.max_length]
            else:
                audio = np.pad(audio, (0, self.max_length - len(audio)), mode='constant')
            
            # Extract features
            inputs = self.feature_extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length
            )
            
            # Process labels
            labels = {
                'speech_text': item.get('transcript', ''),
                'emotion_label': item.get('emotion', 'neutral'),
                'audio_type': item.get('audio_type', 'speech'),
                'tone_features': torch.tensor(item.get('tone_features', [0.5, 0.5, 0.5])),
                'music_genre': item.get('music_genre', 'unknown'),
                'noise_type': item.get('noise_type', 'none')
            }
            
            return {
                'input_values': inputs.input_values.squeeze(),
                'attention_mask': inputs.attention_mask.squeeze(),
                'labels': labels
            }
            
        except Exception as e:
            logger.error(f"Audio loading error {audio_path}: {str(e)}")
            # Return empty data
            return {
                'input_values': torch.zeros(self.max_length),
                'attention_mask': torch.ones(self.max_length),
                'labels': {
                    'speech_text': '',
                    'emotion_label': 'neutral',
                    'audio_type': 'speech',
                    'tone_features': torch.tensor([0.5, 0.5, 0.5]),
                    'music_genre': 'unknown',
                    'noise_type': 'none'
                }
            }


class AudioModelTrainer:
    """Audio model trainer"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize speech recognition model
        self.speech_model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-base-960h"
        ).to(self.device)
        
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        
        # Initialize audio classification model
        self.classification_model = AutoModelForAudioClassification.from_pretrained(
            "superb/hubert-base-superb-ks",
            num_labels=10  # 10 audio types
        ).to(self.device)
        
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "superb/hubert-base-superb-ks"
        )
        
        # Emotion classification head
        self.emotion_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 8)  # 8 emotions
        ).to(self.device)
        
        # Tone analysis head
        self.tone_analyzer = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # Tone features: pitch, intensity, timbre
        ).to(self.device)
        
        # Loss functions
        self.ctc_loss = nn.CTCLoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.emotion_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
    
    
    def load_data(self, data_dir):
        """Load audio data"""
        dataset = AudioDataset(data_dir, self.feature_extractor)
        return dataset
    
    
    def create_data_loader(self, dataset, batch_size=4, shuffle=True):
        """Create data loader"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2  # Fewer workers for audio processing
        )
    
    
    def train_epoch(self, train_loader, optimizer):
        """Single epoch training"""
        self.speech_model.train()
        self.classification_model.train()
        self.emotion_classifier.train()
        self.tone_analyzer.train()
        
        total_ctc_loss = 0
        total_emotion_loss = 0
        total_tone_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_values = batch['input_values'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels']
            
            optimizer.zero_grad()
            
            # Speech recognition forward pass
            try:
                # Process speech text labels
                if all(text.strip() for text in labels['speech_text']):
                    speech_labels = self.processor(
                        labels['speech_text'], 
                        return_tensors="pt",
                        padding=True
                    ).input_ids.to(self.device)
                    speech_outputs = self.speech_model(
                        input_values=input_values,
                        attention_mask=attention_mask,
                        labels=speech_labels
                    )
                    ctc_loss = speech_outputs.loss
                else:
                    # Skip CTC loss if no valid text labels
                    speech_outputs = self.speech_model(
                        input_values=input_values,
                        attention_mask=attention_mask
                    )
                    ctc_loss = torch.tensor(0.0).to(self.device)
            except Exception as e:
                logger.warning(f"CTC loss calculation error: {str(e)}")
                # Skip CTC loss calculation
                speech_outputs = self.speech_model(
                    input_values=input_values,
                    attention_mask=attention_mask
                )
                ctc_loss = torch.tensor(0.0).to(self.device)
            
            # Audio classification and feature extraction
            classification_outputs = self.classification_model(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Extract embeddings for additional tasks
            hidden_states = classification_outputs.hidden_states[-1]
            cls_embeddings = hidden_states[:, 0, :] if hidden_states is not None else torch.zeros(input_values.size(0), 768).to(self.device)
            
            # Emotion analysis
            emotion_logits = self.emotion_classifier(cls_embeddings)
            
            # Tone analysis
            tone_features = self.tone_analyzer(cls_embeddings)
            
            # Calculate various losses
            emotion_labels = torch.tensor([self._emotion_to_idx(e) for e in labels['emotion_label']]).to(self.device)
            emotion_loss = self.emotion_loss(emotion_logits, emotion_labels)
            
            tone_targets = torch.stack([l['tone_features'] for l in labels['tone_features']]).to(self.device) if isinstance(labels['tone_features'], list) else labels['tone_features'].to(self.device)
            tone_loss = self.regression_loss(tone_features, tone_targets)
            
            # Total loss with balanced weights
            total_loss = ctc_loss * 0.5 + emotion_loss * 0.3 + tone_loss * 0.2
            
            total_loss.backward()
            optimizer.step()
            
            total_ctc_loss += ctc_loss.item()
            total_emotion_loss += emotion_loss.item()
            total_tone_loss += tone_loss.item()
        
        return (total_ctc_loss / len(train_loader), 
                total_emotion_loss / len(train_loader),
                total_tone_loss / len(train_loader))
    
    
    def _emotion_to_idx(self, emotion):
        """Emotion label mapping"""
        emotion_mapping = {
            'happy': 0, 'sad': 1, 'angry': 2, 'surprised': 3,
            'fear': 4, 'neutral': 5, 'excited': 6, 'calm': 7
        }
        return emotion_mapping.get(emotion, 5)
    
    
    def evaluate(self, test_loader):
        """Model evaluation"""
        self.speech_model.eval()
        self.classification_model.eval()
        self.emotion_classifier.eval()
        self.tone_analyzer.eval()
        
        total_ctc_loss = 0
        total_emotion_loss = 0
        total_tone_loss = 0
        emotion_accuracy = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_values = batch['input_values'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                
                # Speech recognition
                try:
                    if all(text.strip() for text in labels['speech_text']):
                        speech_labels = self.processor(
                            labels['speech_text'], 
                            return_tensors="pt",
                            padding=True
                        ).input_ids.to(self.device)
                        speech_outputs = self.speech_model(
                            input_values=input_values,
                            attention_mask=attention_mask,
                            labels=speech_labels
                        )
                        ctc_loss = speech_outputs.loss
                    else:
                        speech_outputs = self.speech_model(
                            input_values=input_values,
                            attention_mask=attention_mask
                        )
                        ctc_loss = torch.tensor(0.0).to(self.device)
                except Exception as e:
                    logger.warning(f"CTC loss evaluation error: {str(e)}")
                    speech_outputs = self.speech_model(
                        input_values=input_values,
                        attention_mask=attention_mask
                    )
                    ctc_loss = torch.tensor(0.0).to(self.device)
                
                # Audio classification and feature extraction
                classification_outputs = self.classification_model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Extract embeddings for additional tasks
                hidden_states = classification_outputs.hidden_states[-1]
                cls_embeddings = hidden_states[:, 0, :] if hidden_states is not None else torch.zeros(input_values.size(0), 768).to(self.device)
                emotion_logits = self.emotion_classifier(cls_embeddings)
                tone_features = self.tone_analyzer(cls_embeddings)
                
                # Calculate emotion accuracy
                emotion_labels = torch.tensor([self._emotion_to_idx(e) for e in labels['emotion_label']]).to(self.device)
                emotion_loss = self.emotion_loss(emotion_logits, emotion_labels)
                emotion_preds = emotion_logits.argmax(dim=1)
                emotion_accuracy += (emotion_preds == emotion_labels).sum().item()
                
                # Calculate tone loss
                tone_targets = torch.stack([l['tone_features'] for l in labels['tone_features']]).to(self.device) if isinstance(labels['tone_features'], list) else labels['tone_features'].to(self.device)
                tone_loss = self.regression_loss(tone_features, tone_targets)
                
                total_ctc_loss += ctc_loss.item()
                total_emotion_loss += emotion_loss.item()
                total_tone_loss += tone_loss.item()
        
        avg_ctc_loss = total_ctc_loss / len(test_loader)
        avg_emotion_loss = total_emotion_loss / len(test_loader)
        avg_tone_loss = total_tone_loss / len(test_loader)
        emotion_acc = emotion_accuracy / len(test_loader.dataset)
        
        return avg_ctc_loss, avg_emotion_loss, avg_tone_loss, emotion_acc
    
    
    def save_model(self, path):
        """Save model"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model components that can be serialized
        torch.save({
            'speech_model_state_dict': self.speech_model.state_dict(),
            'classification_model_state_dict': self.classification_model.state_dict(),
            'emotion_classifier_state_dict': self.emotion_classifier.state_dict(),
            'tone_analyzer_state_dict': self.tone_analyzer.state_dict(),
            'config': self.config
        }, path)
        
        # Save processor and feature extractor separately
        processor_path = os.path.join(os.path.dirname(path), 'processor')
        feature_extractor_path = os.path.join(os.path.dirname(path), 'feature_extractor')
        os.makedirs(processor_path, exist_ok=True)
        os.makedirs(feature_extractor_path, exist_ok=True)
        
        self.processor.save_pretrained(processor_path)
        self.feature_extractor.save_pretrained(feature_extractor_path)
        
        logger.info(f"Audio model saved to {path}")
    
    
    def full_training(self, data_dir, epochs=10):
        """Full training pipeline"""
        # Load data
        dataset = self.load_data(data_dir)
        
        # If dataset is empty, create synthetic data
        if len(dataset) == 0:
            logger.warning("No real data found. Creating synthetic training data...")
            self._create_synthetic_data(data_dir)
            dataset = self.load_data(data_dir)
        
        # Split into train and test sets
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        
        # Create data loaders
        train_loader = self.create_data_loader(train_dataset, batch_size=self.config['batch_size'])
        test_loader = self.create_data_loader(test_dataset, batch_size=self.config['batch_size'])
        
        # Setup optimizer
        optimizer = optim.AdamW([
            {'params': self.speech_model.parameters()},
            {'params': self.classification_model.parameters()},
            {'params': self.emotion_classifier.parameters()},
            {'params': self.tone_analyzer.parameters()}
        ], lr=self.config['learning_rate'])
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        
        # Training loop
        best_emotion_acc = 0
        for epoch in range(epochs):
            train_ctc_loss, train_emotion_loss, train_tone_loss = self.train_epoch(train_loader, optimizer)
            test_ctc_loss, test_emotion_loss, test_tone_loss, test_emotion_acc = self.evaluate(test_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train CTC Loss: {train_ctc_loss:.4f} | Train Emotion Loss: {train_emotion_loss:.4f} | Train Tone Loss: {train_tone_loss:.4f} | "
                f"Test CTC Loss: {test_ctc_loss:.4f} | Test Emotion Loss: {test_emotion_loss:.4f} | Test Tone Loss: {test_tone_loss:.4f} | "
                f"Emotion Accuracy: {test_emotion_acc:.4f}"
            )
            
            # Step the scheduler based on emotion loss
            scheduler.step(test_emotion_loss)
            
            # Save best model
            if test_emotion_acc > best_emotion_acc:
                best_emotion_acc = test_emotion_acc
                self.save_model(self.config['model_save_path'])
                logger.info(f"New best model saved with emotion accuracy: {best_emotion_acc:.4f}")
        
        logger.info("Audio model training completed")
    
    def _create_synthetic_data(self, data_dir):
        """Create synthetic training data when real data is not available"""
        os.makedirs(data_dir, exist_ok=True)
        
        # Create synthetic audio data with emotion labels
        emotions = ['happy', 'sad', 'angry', 'surprised', 'fear', 'neutral', 'excited', 'calm']
        audio_types = ['speech', 'music', 'noise', 'tone', 'effects']
        
        # Create dummy audio files directory
        audio_dir = os.path.join(data_dir, 'audio_files')
        os.makedirs(audio_dir, exist_ok=True)
        
        # Generate synthetic data for each audio type
        for audio_type in audio_types:
            synthetic_data = []
            for i in range(30):  # Generate 30 samples per type
                # Create a dummy audio file
                dummy_audio_path = os.path.join(audio_dir, f'dummy_{audio_type}_{i}.wav')
                if not os.path.exists(dummy_audio_path):
                    # Create a dummy audio using numpy and soundfile
                    if audio_type == 'speech':
                        # Generate speech-like noise
                        audio = np.random.normal(0, 0.01, 16000 * 3).astype(np.float32)  # 3 seconds
                    elif audio_type == 'music':
                        # Generate music-like noise
                        t = np.linspace(0, 3, 16000 * 3)
                        audio = np.sin(2 * np.pi * 440 * t + 0.5 * np.sin(2 * np.pi * 2 * t)).astype(np.float32) * 0.1
                    else:
                        # Generate random noise for other types
                        audio = np.random.normal(0, 0.02, 16000 * 3).astype(np.float32)
                    
                    sf.write(dummy_audio_path, audio, 16000)
                
                # Add to dataset
                synthetic_data.append({
                    'audio_path': dummy_audio_path,
                    'transcript': f"This is a dummy {audio_type} sample number {i}",
                    'emotion': emotions[i % len(emotions)],
                    'audio_type': audio_type,
                    'tone_features': [0.5 + 0.3 * np.random.rand(), 0.5 + 0.3 * np.random.rand(), 0.5 + 0.3 * np.random.rand()],
                    'music_genre': 'unknown',
                    'noise_type': 'none',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Save synthetic data to JSON file
            output_file = os.path.join(data_dir, f'audio_{audio_type}.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(synthetic_data, f, indent=2)
            
            logger.info(f"Created synthetic {audio_type} data: {len(synthetic_data)} samples")

# Example configuration
if __name__ == "__main__":
    config = {
        'batch_size': 2,  # Smaller batch size for audio processing
        'learning_rate': 3e-5,
        'model_save_path': 'models/audio_model.pth'
    }
    
    # Audio data directory
    audio_data_dir = 'data/audio/'
    
    # Create data directory if it doesn't exist
    os.makedirs(audio_data_dir, exist_ok=True)
    
    trainer = AudioModelTrainer(config)
    trainer.full_training(audio_data_dir, epochs=10)
