"""
=================================================================================
IMAGE CAPTION GENERATION - FLICKR30K
ULTRA FINAL - FIXED SEQUENCE RETURN TYPE
=================================================================================
"""

import os
import sys
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import warnings
warnings.filterwarnings('ignore')

tf.get_logger().setLevel('ERROR')

print("\n" + "="*80)
print("IMAGE CAPTION GENERATION - ULTRA FINAL VERSION")
print("="*80)
print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {len(tf.config.list_physical_devices('GPU'))} GPU(s)")
print("="*80 + "\n")

class Config:
    IMAGE_FEATURES_PATH = '/kaggle/input/inception-v3-features-pkl/inception_v3_features.pkl'
    CAPTIONS_PATH = '/kaggle/input/flickr-image-dataset/flickr30k_images/results.csv'
    OUTPUT_DIR = '/kaggle/working/'

    EMBEDDING_DIM = 256
    LSTM_UNITS = 256
    DROPOUT_RATE = 0.5
    VOCAB_SIZE = 10000
    MAX_CAPTION_LENGTH = 40
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.001
    PATIENCE = 5
    FEATURE_DIM = 2048

config = Config()

def verify_paths():
    print("\n" + "="*80)
    print("VERIFYING PATHS")
    print("="*80)

    features_path = None
    captions_path = None

    if os.path.exists(config.IMAGE_FEATURES_PATH):
        features_path = config.IMAGE_FEATURES_PATH
        print(f"✓ Features: {features_path}")
    else:
        for root, dirs, files in os.walk('/kaggle/input/'):
            for file in files:
                if file.endswith('.pkl'):
                    features_path = os.path.join(root, file)
                    print(f"✓ Found features: {features_path}")
                    break
            if features_path:
                break

    if os.path.exists(config.CAPTIONS_PATH):
        captions_path = config.CAPTIONS_PATH
        print(f"✓ Captions: {captions_path}")
    else:
        for root, dirs, files in os.walk('/kaggle/input/'):
            for file in files:
                if file.endswith('.csv'):
                    captions_path = os.path.join(root, file)
                    print(f"✓ Found captions: {captions_path}")
                    break
            if captions_path:
                break

    if features_path is None:
        print("❌ Features not found!")
        return None, None

    if captions_path is None:
        print("❌ Captions not found!")
        return None, None

    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print(f"✓ Output: {config.OUTPUT_DIR}")

    return features_path, captions_path

def load_features(features_path):
    print("\n" + "="*80)
    print("LOADING FEATURES")
    print("="*80)

    try:
        with open(features_path, 'rb') as f:
            features = pickle.load(f)

        if not isinstance(features, dict) or len(features) == 0:
            print(f"❌ Invalid features!")
            return None, None

        print(f"✓ Loaded: {len(features)} images")

        sample_key = next(iter(features.keys()))
        sample_feature = features[sample_key]

        if not isinstance(sample_feature, np.ndarray):
            print(f"❌ Feature must be ndarray!")
            return None, None

        if len(sample_feature.shape) == 2:
            feature_dim = sample_feature[0].shape[0]
        elif len(sample_feature.shape) == 1:
            feature_dim = sample_feature.shape[0]
        else:
            print(f"❌ Unexpected shape!")
            return None, None

        print(f"✓ Feature dimension: {feature_dim}")
        return features, feature_dim

    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

def load_captions(captions_path):
    print("\n" + "="*80)
    print("LOADING CAPTIONS")
    print("="*80)

    try:
        print(f"Loading: {captions_path}")

        captions_dict = {}
        processed = 0
        skipped = 0

        with open(captions_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f):
                try:
                    line = line.strip()

                    if not line or line_num == 0 or 'image_name' in line.lower():
                        if line_num == 0:
                            print(f"  Skipping header")
                        skipped += 1
                        continue

                    parts = line.split('|', 2)
                    if len(parts) < 3:
                        skipped += 1
                        continue

                    img_id = parts[0].strip()
                    if not img_id or '.' in img_id:
                        img_id = img_id.rsplit('.', 1)[0] if '.' in img_id else img_id

                    caption = parts[2].strip()

                    if not caption or len(caption) < 2 or caption.lower() in ['nan', 'none', 'comment']:
                        skipped += 1
                        continue

                    caption = caption.lower()
                    caption = ''.join([c if (c.isalpha() or c.isspace()) else '' for c in caption])
                    caption = ' '.join(caption.split())

                    if len(caption) < 3:
                        skipped += 1
                        continue

                    caption = 'startseq ' + caption + ' endseq'

                    if img_id not in captions_dict:
                        captions_dict[img_id] = []

                    captions_dict[img_id].append(caption)
                    processed += 1

                    if line_num < 6:
                        print(f"  {caption[:60]}...")

                except:
                    continue

        print(f"\n✓ Processed: {processed}")
        print(f"✓ Images: {len(captions_dict)}")

        if len(captions_dict) == 0:
            print("❌ No captions loaded!")
            return {}

        first_key = next(iter(captions_dict.keys()))
        print(f"✓ Sample: {captions_dict[first_key][0][:70]}...")

        return captions_dict

    except Exception as e:
        print(f"❌ Error: {e}")
        return {}

def create_tokenizer(captions_dict, vocab_size):
    print("\n" + "="*80)
    print("CREATING TOKENIZER")
    print("="*80)

    if not captions_dict:
        print("❌ Empty captions!")
        return None, 0

    all_captions = []
    for caps_list in captions_dict.values():
        all_captions.extend(caps_list)

    if len(all_captions) == 0:
        print("❌ No captions!")
        return None, 0

    print(f"✓ Captions: {len(all_captions)}")

    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<UNK>')
    tokenizer.fit_on_texts(all_captions)

    actual_vocab = len(tokenizer.word_index) + 1
    print(f"✓ Vocabulary: {actual_vocab}")

    return tokenizer, actual_vocab

def calculate_max_length(captions_dict):
    if not captions_dict:
        return 40
    max_len = max(len(c.split()) for caps in captions_dict.values() for c in caps)
    return min(max(max_len, 10), 50)

# CRITICAL FIX: Return TUPLE not LIST from __getitem__
class DataGenerator(keras.utils.Sequence):

    def __init__(self, img_ids, features, captions, tokenizer, max_len, vocab_sz, batch_sz):
        self.img_ids = img_ids
        self.features = features
        self.captions = captions
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.vocab_sz = vocab_sz
        self.batch_sz = max(1, batch_sz)
        self.samples = self._create_samples()

        if len(self.samples) == 0:
            raise ValueError("No samples!")

    def _create_samples(self):
        samples = []
        for img_id in self.img_ids:
            if img_id in self.features and img_id in self.captions:
                for caption in self.captions[img_id]:
                    samples.append((img_id, caption))
        return samples

    def __len__(self):
        return max(1, int(np.ceil(len(self.samples) / self.batch_sz)))

    def __getitem__(self, idx):
        try:
            batch_start = idx * self.batch_sz
            batch_end = min(batch_start + self.batch_sz, len(self.samples))
            batch = self.samples[batch_start:batch_end]

            if len(batch) == 0:
                raise ValueError(f"Empty batch!")

            X1, X2, y = [], [], []

            for img_id, caption in batch:
                try:
                    feature = self.features[img_id]
                    if isinstance(feature, np.ndarray) and len(feature.shape) == 2:
                        feature = feature[0]

                    seq = self.tokenizer.texts_to_sequences([caption])[0]
                    if len(seq) < 2:
                        continue

                    for i in range(1, len(seq)):
                        in_seq = pad_sequences([seq[:i]], maxlen=self.max_len, padding='post')[0]
                        out_seq = keras.utils.to_categorical([seq[i]], num_classes=self.vocab_sz)[0]

                        X1.append(feature)
                        X2.append(in_seq)
                        y.append(out_seq)

                except:
                    continue

            if len(X1) == 0:
                first_feature = self.features[next(iter(self.features.keys()))]
                if len(first_feature.shape) == 2:
                    first_feature = first_feature[0]
                X1 = [np.zeros_like(first_feature)]
                X2 = [np.zeros(self.max_len, dtype=np.int32)]
                y = [np.zeros(self.vocab_sz, dtype=np.float32)]

            # CRITICAL FIX: Return TUPLE not LIST
            # Before: return [np.array(X1, dtype=np.float32), np.array(X2, dtype=np.int32)], np.array(y, dtype=np.float32)
            # After: 
            X = (np.array(X1, dtype=np.float32), np.array(X2, dtype=np.int32))
            y = np.array(y, dtype=np.float32)
            return X, y

        except Exception as e:
            print(f"Error in batch: {e}")
            raise

def build_model(vocab_sz, max_len, feat_dim, emb_dim, lstm_units, drop_rate):
    print("\n" + "="*80)
    print("BUILDING MODEL")
    print("="*80)

    try:
        img_in = Input(shape=(feat_dim,), name='img_input')
        img_dense = Dense(emb_dim, activation='relu', name='img_dense')(img_in)
        img_bn = BatchNormalization(name='img_bn')(img_dense)
        img_drop = Dropout(drop_rate, name='img_drop')(img_bn)

        cap_in = Input(shape=(max_len,), name='cap_input')
        cap_emb = Embedding(vocab_sz, emb_dim, mask_zero=True, name='cap_emb')(cap_in)
        cap_drop1 = Dropout(drop_rate, name='cap_drop1')(cap_emb)
        cap_lstm = LSTM(lstm_units, name='cap_lstm')(cap_drop1)
        cap_drop2 = Dropout(drop_rate, name='cap_drop2')(cap_lstm)

        merged = Add(name='merge')([img_drop, cap_drop2])
        dec_dense = Dense(emb_dim, activation='relu', name='dec_dense')(merged)
        dec_bn = BatchNormalization(name='dec_bn')(dec_dense)
        dec_drop = Dropout(drop_rate, name='dec_drop')(dec_bn)
        output = Dense(vocab_sz, activation='softmax', name='output')(dec_drop)

        model = Model(inputs=[img_in, cap_in], outputs=output)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
            metrics=['accuracy']
        )

        print(f"✓ Parameters: {model.count_params():,}")

        return model

    except Exception as e:
        print(f"❌ Error: {e}")
        raise

def create_callbacks():
    print("\n" + "="*80)
    print("SETTING UP CALLBACKS")
    print("="*80)

    callbacks = []

    best_model_path = os.path.join(config.OUTPUT_DIR, 'best_model.keras')
    callbacks.append(ModelCheckpoint(filepath=best_model_path, monitor='val_loss', save_best_only=True, verbose=1))
    print(f"✓ ModelCheckpoint")

    callbacks.append(EarlyStopping(monitor='val_loss', patience=config.PATIENCE, restore_best_weights=True, verbose=1))
    print(f"✓ EarlyStopping")

    callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1))
    print(f"✓ ReduceLROnPlateau")

    return callbacks

def train_model(model, train_gen, val_gen, callbacks):
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Train: {len(train_gen.samples)} | Val: {len(val_gen.samples)} | Batch: {config.BATCH_SIZE}")
    print("="*80)

    try:
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config.EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        return history
    except Exception as e:
        print(f"❌ Training error: {e}")
        raise

def generate_caption(model, tokenizer, feature, max_len):
    try:
        caption = 'startseq'
        for _ in range(max_len):
            seq = tokenizer.texts_to_sequences([caption])[0]
            seq = pad_sequences([seq], maxlen=max_len, padding='post')
            pred = model.predict([feature.reshape(1, -1), seq], verbose=0)
            pred_id = np.argmax(pred)

            pred_word = None
            for word, idx in tokenizer.word_index.items():
                if idx == pred_id:
                    pred_word = word
                    break

            if pred_word is None or pred_word == 'endseq':
                break
            caption += ' ' + pred_word

        return caption.replace('startseq', '').strip()
    except:
        return ""

def evaluate_model(model, test_ids, features, captions, tokenizer, max_len):
    print("\n" + "="*80)
    print("EVALUATING MODEL")
    print("="*80)

    try:
        actual = []
        predicted = []

        for img_id in test_ids[:min(100, len(test_ids))]:
            if img_id not in features or img_id not in captions:
                continue

            actual_captions = [c.replace('startseq', '').replace('endseq', '').strip().split() for c in captions[img_id]]
            feature = features[img_id]
            if isinstance(feature, np.ndarray) and len(feature.shape) == 2:
                feature = feature[0]

            pred_cap = generate_caption(model, tokenizer, feature, max_len)
            if pred_cap:
                actual.append(actual_captions)
                predicted.append(pred_cap.split())

        if len(actual) == 0:
            print("⚠ No predictions")
            return {'bleu1': 0, 'bleu2': 0, 'bleu3': 0, 'bleu4': 0}

        smooth_fn = SmoothingFunction().method1
        bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0), smoothing_function=smooth_fn)
        bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn)
        bleu3 = corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth_fn)
        bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)

        print(f"BLEU: 1={bleu1:.4f} 2={bleu2:.4f} 3={bleu3:.4f} 4={bleu4:.4f}")

        return {'bleu1': bleu1, 'bleu2': bleu2, 'bleu3': bleu3, 'bleu4': bleu4}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {'bleu1': 0, 'bleu2': 0, 'bleu3': 0, 'bleu4': 0}

def main():
    print("\n" + "="*80)
    print("🚀 IMAGE CAPTION GENERATION")
    print("="*80)

    try:
        features_path, captions_path = verify_paths()
        if features_path is None or captions_path is None:
            return

        features, feat_dim = load_features(features_path)
        if features is None:
            return

        config.FEATURE_DIM = feat_dim
        captions = load_captions(captions_path)
        if not captions:
            return

        valid_ids = list(set(features.keys()) & set(captions.keys()))
        print(f"\n✓ Valid: {len(valid_ids)}")

        if len(valid_ids) < 50:
            print("❌ Too few images")
            return

        tokenizer, vocab_size = create_tokenizer(captions, config.VOCAB_SIZE)
        if tokenizer is None:
            return

        config.VOCAB_SIZE = vocab_size
        max_len = calculate_max_length(captions)
        config.MAX_CAPTION_LENGTH = max_len

        train_ids, test_ids = train_test_split(valid_ids, test_size=0.2, random_state=42)
        train_ids, val_ids = train_test_split(train_ids, test_size=0.1, random_state=42)

        print(f"✓ Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

        train_gen = DataGenerator(train_ids, features, captions, tokenizer, max_len, config.VOCAB_SIZE, config.BATCH_SIZE)
        val_gen = DataGenerator(val_ids, features, captions, tokenizer, max_len, config.VOCAB_SIZE, config.BATCH_SIZE)

        model = build_model(config.VOCAB_SIZE, max_len, feat_dim, config.EMBEDDING_DIM, config.LSTM_UNITS, config.DROPOUT_RATE)
        callbacks = create_callbacks()
        history = train_model(model, train_gen, val_gen, callbacks)

        final_model_path = os.path.join(config.OUTPUT_DIR, 'final_model.keras')
        model.save(final_model_path)
        print(f"✓ Model saved")

        tokenizer_path = os.path.join(config.OUTPUT_DIR, 'tokenizer.pkl')
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        print(f"✓ Tokenizer saved")

        bleu = evaluate_model(model, test_ids, features, captions, tokenizer, max_len)

        results_file = os.path.join(config.OUTPUT_DIR, 'evaluation_results.txt')
        with open(results_file, 'w') as f:
            f.write("RESULTS\n")
            f.write(f"Images: {len(valid_ids)}\n")
            f.write(f"BLEU: 1={bleu['bleu1']:.4f} 2={bleu['bleu2']:.4f} 3={bleu['bleu3']:.4f} 4={bleu['bleu4']:.4f}\n")

        print("\n" + "="*80)
        print("✅ COMPLETE!")
        print("="*80)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
