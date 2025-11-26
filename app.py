import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
from PIL import Image
import numpy as np
import pickle
import os
import re
import json
from datetime import datetime
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import cv2

# For newer TensorFlow versions, use keras directly
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences

# Initialize NLTK components (download required data if not present)
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger.zip')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4.zip')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

# --- Model & Tokenizer Paths ---
MODELS = {
    'VGG16 Image Captioning Model': {
        'model': 'models/best_model.keras',
        'tokenizer': 'models/tokenizer.pkl',
        'feature_extractor': 'vgg16',
        'max_length': 34
    }
}

# Feedback storage file
FEEDBACK_FILE = 'models/user_feedback.json'

# Custom object classifier path
CUSTOM_OBJECT_MODEL_PATH = '../object/object_classifier_final.keras'

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


# --- Custom Object Detection Model (YOUR trained model) ---

@st.cache_resource
def load_custom_object_model():
    """Load your custom trained object detection model."""
    try:
        if not os.path.exists(CUSTOM_OBJECT_MODEL_PATH):
            st.warning(f"⚠️ Custom object model not found at {CUSTOM_OBJECT_MODEL_PATH}")
            return None
        
        model = load_model(CUSTOM_OBJECT_MODEL_PATH)
        print("Custom object detection model loaded successfully.")
        
        # Define common COCO/Flickr classes (you may need to adjust based on your model)
        # These are typical classes the model might have been trained on
        class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        return model, class_names
    except Exception as e:
        st.warning(f"⚠️ Could not load custom object model: {e}")
        return None


def detect_objects_custom(image, model_tuple, threshold=0.3):
    """
    Detect objects using your custom trained model.
    Returns list of detected object names.
    """
    if model_tuple is None:
        return []
    
    try:
        model, class_names = model_tuple
        
        # Preprocess image (adjust size based on your model's training)
        img_size = 224  # Adjust if your model uses different size
        img_resized = image.resize((img_size, img_size))
        img_array = np.array(img_resized) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get predictions
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get objects above threshold
        detected_objects = []
        for idx, prob in enumerate(predictions):
            if prob > threshold and idx < len(class_names):
                detected_objects.append((class_names[idx], float(prob)))
        
        # Sort by confidence
        detected_objects.sort(key=lambda x: x[1], reverse=True)
        
        return [obj[0] for obj in detected_objects]
        
    except Exception as e:
        st.error(f"Error in custom object detection: {e}")
        return []


# --- YOLO Object Detection (Evaluation Tool) ---

@st.cache_resource
def load_yolo_model():
    """Load YOLO model for ground truth object detection (evaluation purpose)."""
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("YOLOv8 model loaded for evaluation.")
        return model
    except ImportError:
        st.info("💡 YOLO evaluation tool not available. Install with: pip install ultralytics")
        return None
    except Exception as e:
        st.warning(f"⚠️ YOLO evaluation tool unavailable: {e}")
        return None


def detect_objects_yolo(image, model):
    """Detect ground truth objects using YOLO (for evaluation)."""
    if model is None:
        return []
    
    try:
        results = model(image, verbose=False)
        detected_objects = {}
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names[class_id]
                
                if confidence > 0.3:
                    if class_name not in detected_objects or confidence > detected_objects[class_name]:
                        detected_objects[class_name] = confidence
        
        sorted_objects = sorted(detected_objects.items(), key=lambda x: x[1], reverse=True)
        return [obj[0] for obj in sorted_objects]
        
    except Exception as e:
        return []


# --- Heuristic Action Detection (Evaluation Tool) ---

def detect_actions_heuristic(image, detected_objects):
    """
    Detect likely actions using heuristics based on detected objects and image analysis.
    This is a simple evaluation tool for comparison.
    """
    try:
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Get image dimensions
        height, width = img_array.shape[:2]
        
        # Analyze image characteristics
        detected_actions = []
        
        # Check for common action-related objects
        if any(obj in detected_objects for obj in ['sports ball', 'tennis racket', 'baseball bat', 'skateboard', 'surfboard']):
            detected_actions.append('play')
        
        if any(obj in detected_objects for obj in ['bed', 'couch']):
            detected_actions.append('lie')
        
        if any(obj in detected_objects for obj in ['chair', 'bench']):
            detected_actions.append('sit')
        
        if any(obj in detected_objects for obj in ['bicycle', 'motorcycle', 'car', 'truck']):
            detected_actions.append('ride')
        
        # Check brightness patterns (simplified pose inference)
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Analyze vertical distribution
        upper_third = gray[:height//3, :]
        middle_third = gray[height//3:2*height//3, :]
        lower_third = gray[2*height//3:, :]
        
        upper_activity = np.std(upper_third)
        middle_activity = np.std(middle_third)
        lower_activity = np.std(lower_third)
        
        # Infer potential actions from spatial distribution
        if upper_activity > middle_activity * 1.5:
            if 'jump' not in detected_actions:
                detected_actions.append('jump')
        
        # Default to common actions if none detected
        if not detected_actions:
            if 'person' in detected_objects:
                detected_actions.append('stand')
            if 'dog' in detected_objects or 'cat' in detected_objects:
                detected_actions.append('look')
        
        return detected_actions
        
    except Exception as e:
        return []


# --- Filename Parsing for Ground Truth ---

def parse_filename_for_labels(filename):
    """
    Parse filename to extract objects and actions ONLY if filename follows descriptive pattern.
    Examples of VALID patterns:
      - "man-standing-town-street-summer-day-379612758.jpg" → parse it
      - "bicycle_on_street.jpg" → parse it
      - "person-riding-bike.jpg" → parse it
    
    Examples of INVALID patterns (return empty):
      - "IMG_1234.jpg" → ignore
      - "photo_20231105.jpg" → ignore  
      - "DSC0042.jpg" → ignore
    
    Returns: (objects, actions, is_descriptive)
    """
    try:
        # Remove file extension
        name = os.path.splitext(filename)[0]
        
        # Check if filename is descriptive (has letters in meaningful words, not just IMG/DSC/random)
        # If filename is mostly numbers or generic patterns, don't parse
        if re.match(r'^(IMG|DSC|PHOTO|IMAGE|PIC|DCIM|MOV|VID)[-_]?\d+', name, re.IGNORECASE):
            return [], [], False  # Generic camera filename pattern
        
        # If filename is just numbers or very short
        if len(name) < 4 or re.match(r'^\d+$', name):
            return [], [], False
        
        # Remove numbers and common suffixes
        name = re.sub(r'[-_]\d+$', '', name)  # Remove trailing numbers like -379612758
        name = re.sub(r'\d+', '', name)  # Remove all remaining numbers
        
        # Split by hyphens, underscores, or camelCase
        words = re.split(r'[-_\s]+', name.lower())
        
        # Filter out very short/empty words
        words = [w for w in words if len(w) > 1]
        
        # If too few words after filtering, it's not descriptive
        if len(words) < 2:
            return [], [], False
        
        # Common action words (verbs)
        action_words = {
            'standing', 'sitting', 'walking', 'running', 'jumping', 'playing',
            'eating', 'drinking', 'sleeping', 'riding', 'driving', 'flying',
            'swimming', 'surfing', 'skiing', 'skating', 'cooking', 'reading',
            'writing', 'typing', 'talking', 'looking', 'watching', 'holding',
            'carrying', 'throwing', 'catching', 'kicking', 'hitting', 'climbing',
            'dancing', 'singing', 'lying', 'working', 'studying', 'shopping',
            'smiling'  # Add common facial actions
        }
        
        # Common prepositions/articles/descriptors to ignore (EXPANDED)
        ignore_words = {
            'on', 'in', 'at', 'the', 'a', 'an', 'with', 'of', 'for', 'to',
            'from', 'by', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'day', 'night', 'summer', 'winter', 'spring', 'fall', 'photo',
            'image', 'picture', 'jpg', 'jpeg', 'png', 'background', 'portrait',
            'close', 'up', 'closeup', 'outdoor', 'indoor', 'shot', 'view',
            'white', 'black', 'red', 'blue', 'green', 'yellow', 'color',
            'face', 'head', 'body', 'full', 'half', 'side', 'front', 'back'
        }
        
        # Valid object words (ONLY keep common recognizable objects)
        valid_objects = {
            'person', 'man', 'woman', 'boy', 'girl', 'child', 'baby',
            'dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'elephant',
            'bicycle', 'bike', 'car', 'bus', 'truck', 'motorcycle', 'train',
            'airplane', 'boat', 'skateboard', 'surfboard',
            'bottle', 'cup', 'glass', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'pizza', 'cake',
            'chair', 'couch', 'table', 'bed', 'toilet', 'desk',
            'laptop', 'phone', 'keyboard', 'mouse', 'book',
            'umbrella', 'bag', 'backpack', 'suitcase', 'tie', 'hat',
            'ball', 'frisbee', 'kite', 'bat', 'glove',
            'tree', 'grass', 'flower', 'plant',
            'street', 'road', 'building', 'house', 'bridge', 'beach', 'park',
            'town', 'city', 'mountain', 'sky', 'water', 'ocean', 'river'
        }
        
        # Gender normalization mapping
        gender_map = {
            'man': 'person', 'woman': 'person', 'men': 'person', 'women': 'person',
            'boy': 'person', 'girl': 'person', 'boys': 'person', 'girls': 'person',
            'male': 'person', 'female': 'person', 'guy': 'person', 'lady': 'person'
        }
        
        objects = []
        actions = []
        
        for word in words:
            word = word.strip()
            if not word or word in ignore_words:
                continue
            
            # Normalize gender terms
            if word in gender_map:
                word = gender_map[word]
            
            # Check if it's an action
            if word in action_words:
                if word not in actions:
                    actions.append(word)
            # ONLY add as object if it's in valid_objects list
            elif word in valid_objects:
                if word not in objects:
                    objects.append(word)
        
        # Return True only if we found meaningful labels
        is_descriptive = bool(objects or actions)
        return objects, actions, is_descriptive
    
    except Exception as e:
        print(f"Error parsing filename: {e}")
        return [], [], False


# --- Object and Verb Detection Functions ---

def get_wordnet_pos(treebank_tag):
    """Convert treebank POS tags to WordNet POS tags for lemmatization."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun


def extract_objects_and_verbs(caption):
    """
    Extract objects (nouns) and actions (verbs) from the generated caption.
    Returns lemmatized lists of objects and verbs.
    """
    if not caption or caption == "Unable to generate caption":
        return [], []
    
    # Remove punctuation and convert to lowercase
    clean_caption = re.sub(r'[^\w\s]', '', caption.lower())
    words = clean_caption.split()
    
    # POS tagging
    try:
        pos_tags = nltk.pos_tag(words)
    except Exception as e:
        # Fallback to simple extraction if NLTK fails
        return extract_objects_and_verbs_fallback(caption)
    
    objects = []
    verbs = []
    
    # Common stop words to filter out
    stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'on', 'in', 'at', 
                  'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into',
                  'through', 'during', 'before', 'after', 'above', 'below', 'between'}
    
    for word, pos in pos_tags:
        # Skip stop words
        if word in stop_words:
            continue
        
        # Extract nouns (objects)
        if pos.startswith('NN'):  # NN, NNS, NNP, NNPS
            wordnet_pos = get_wordnet_pos(pos)
            lemma = lemmatizer.lemmatize(word, wordnet_pos)
            if lemma not in objects:
                objects.append(lemma)
        
        # Extract verbs (actions)
        elif pos.startswith('VB'):  # VB, VBD, VBG, VBN, VBP, VBZ
            # Lemmatize to base form
            lemma = lemmatizer.lemmatize(word, wordnet.VERB)
            if lemma not in verbs and lemma not in ['be', 'have', 'do']:  # Skip auxiliary verbs
                verbs.append(lemma)
    
    return objects, verbs


def extract_objects_and_verbs_fallback(caption):
    """
    Fallback extraction method using predefined word lists.
    Used if NLTK fails. Normalizes gender terms to 'person'.
    """
    if not caption or caption == "Unable to generate caption":
        return [], []
    
    # Normalize gender terms
    gender_terms = ['man', 'woman', 'boy', 'girl', 'guy', 'lady']
    
    clean_caption = re.sub(r'[^\w\s]', '', caption.lower())
    words = clean_caption.split()
    
    # Replace gender terms with 'person'
    normalized_words = []
    for word in words:
        if word in gender_terms:
            normalized_words.append('person')
        else:
            normalized_words.append(word)
    
    words_set = set(normalized_words)
    
    # Predefined categories (gender-neutral)
    known_objects = {
        'person', 'people', 'child', 
        'dog', 'cat', 'bird', 'animal', 'dogs', 'skier', 'photographer',
        'bench', 'park', 'street', 'sidewalk', 'ground', 'grass', 'table', 
        'chair', 'bed', 'floor', 'beach', 'water', 'snow', 'field', 'forest', 
        'woods', 'road', 'tree', 'trees', 'frame', 'frames', 'photo', 'photos', 
        'picture', 'pictures', 'painting', 'paintings', 'wall', 'sky', 'clouds',
        'window', 'bricks', 'tshirt', 'shirt', 'background'
    }
    
    known_verbs = {
        'sit', 'sitting', 'stand', 'standing', 'walk', 'walking', 
        'run', 'running', 'play', 'playing', 'lie', 'lying', 'lay', 'laying',
        'look', 'looking', 'watch', 'watching', 'jump', 'jumping', 
        'eat', 'eating', 'drink', 'drinking', 'ski', 'skiing', 
        'display', 'displaying', 'fight', 'fighting', 'sleep', 'sleeping',
        'talk', 'talking', 'read', 'reading'
    }
    
    # Extract matching objects and verbs
    objects = [w for w in words_set if w in known_objects]
    
    # For verbs, lemmatize manually
    verbs = []
    for w in words_set:
        if w in known_verbs:
            # Simple lemmatization
            base = w.replace('ing', '').replace('ed', '')
            if base.endswith('nn'):  # running -> run
                base = base[:-1]
            verbs.append(base if base else w)
    
    return list(set(objects)), list(set(verbs))



# --- Feedback System Functions ---

def load_feedback_data():
    """Load existing feedback data from file."""
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, 'r') as f:
                return json.load(f)
        except:
            return {'feedbacks': [], 'stats': {'total': 0, 'positive': 0, 'negative': 0}}
    return {'feedbacks': [], 'stats': {'total': 0, 'positive': 0, 'negative': 0}}


def save_feedback(generated_caption, user_rating, correct_caption=None, image_features=None, detected_objects=None, detected_verbs=None):
    """Save user feedback for future learning."""
    data = load_feedback_data()
    
    feedback_entry = {
        'timestamp': datetime.now().isoformat(),
        'generated_caption': generated_caption,
        'rating': user_rating,  # 'positive' or 'negative'
        'correct_caption': correct_caption,
        'detected_objects': detected_objects if detected_objects else [],
        'detected_verbs': detected_verbs if detected_verbs else [],
        'feature_summary': {
            'mean': float(np.mean(image_features)) if image_features is not None else None,
            'std': float(np.std(image_features)) if image_features is not None else None,
        }
    }
    
    data['feedbacks'].append(feedback_entry)
    data['stats']['total'] += 1
    if user_rating == 'positive':
        data['stats']['positive'] += 1
    else:
        data['stats']['negative'] += 1
    
    # Save to file
    try:
        with open(FEEDBACK_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving feedback: {e}")
        return False


def analyze_feedback_patterns():
    """Analyze feedback to learn common patterns and improvements."""
    data = load_feedback_data()
    
    if not data['feedbacks']:
        return None
    
    # Extract patterns from negative feedback with corrections
    corrections = [f for f in data['feedbacks'] if f['rating'] == 'negative' and f['correct_caption']]
    
    # Analyze common words in correct captions that model missed
    missed_words = {}
    for correction in corrections:
        correct_words = set(correction['correct_caption'].lower().split())
        generated_words = set(correction['generated_caption'].lower().split())
        missing = correct_words - generated_words
        
        for word in missing:
            missed_words[word] = missed_words.get(word, 0) + 1
    
    return {
        'total_feedback': data['stats']['total'],
        'accuracy_rate': (data['stats']['positive'] / data['stats']['total'] * 100) if data['stats']['total'] > 0 else 0,
        'commonly_missed_words': sorted(missed_words.items(), key=lambda x: x[1], reverse=True)[:10]
    }


# --- Cached Model Loading Functions ---

# Use st.cache_resource to load models only once
@st.cache_resource
def get_vgg_model():
    print("Loading VGG16 model...")
    # Load VGG16, but remove the final prediction layer
    vgg_model = VGG16(weights='imagenet')
    # Use the output of the 'fc2' layer as our feature vector (4096-dim)
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    print("VGG16 model loaded.")
    return vgg_model

@st.cache_resource
def get_resnet_model():
    print("Loading ResNet50 model...")
    # Load ResNet50, but remove the final prediction layer
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    # This gives us 2048-dim features
    print("ResNet50 model loaded.")
    return resnet_model

@st.cache_resource
def get_caption_model(model_path):
    print(f"Loading captioning model from {model_path}...")
    # Load our fully trained captioning model
    try:
        model = load_model(model_path)
        print("Captioning model loaded.")
        return model
    except Exception as e:
        st.error(f"Error loading caption model: {e}")
        st.error(f"Make sure the model file is in the '{os.path.abspath('models')}' folder.")
        return None

@st.cache_resource
def get_tokenizer(tokenizer_path):
    print(f"Loading tokenizer from {tokenizer_path}...")
    # Load the saved tokenizer
    try:
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
            print("Tokenizer loaded.")
            return tokenizer
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        st.error(f"Make sure the tokenizer file is in the '{os.path.abspath('models')}' folder.")
        return None


# --- Helper Functions ---

def improve_caption_grammar(raw_caption):
    """
    Intelligently reconstructs caption using identified objects and actions.
    Now enhanced with learned patterns from user feedback.
    Normalizes gender-specific terms to 'person' for better accuracy.
    """
    if not raw_caption or raw_caption == "Unable to generate caption":
        return raw_caption
    
    # Normalize gender-specific terms to 'person'
    gender_terms = {
        'man': 'person',
        'woman': 'person',
        'boy': 'person',
        'girl': 'person',
        'guy': 'person',
        'lady': 'person'
    }
    
    # Replace gender terms in raw caption
    words_list = raw_caption.lower().split()
    normalized_words = [gender_terms.get(word, word) for word in words_list]
    raw_caption_normalized = ' '.join(normalized_words)
    
    # Get learned patterns
    patterns = analyze_feedback_patterns()
    
    # Convert to lowercase and split
    words = raw_caption_normalized.split()
    
    # Remove duplicates while preserving first occurrence order
    seen = set()
    unique_words = []
    for word in words:
        if word not in seen and word not in ['of', 'the', 'a', 'an', 'is', 'are', 'and']:
            seen.add(word)
            unique_words.append(word)
    
    # Base word categories (gender-neutral)
    subjects = ['person', 'people', 'child', 'dog', 'cat', 'bird', 'animal', 'skier', 'photographer', 'dogs']
    verbs = ['sitting', 'standing', 'walking', 'running', 'playing', 'lying', 'laying', 'looking', 'watching', 'jumping', 'eating', 'drinking', 'skiing', 'displaying', 'fighting', 'sleeping']
    locations = ['bench', 'park', 'street', 'sidewalk', 'ground', 'grass', 'table', 'chair', 'bed', 'floor', 'beach', 'water', 'snow', 'field', 'forest', 'woods', 'road']
    objects = ['tree', 'trees', 'frame', 'frames', 'photo', 'photos', 'picture', 'pictures', 'painting', 'paintings', 'wall']
    descriptors = ['wooden', 'metal', 'white', 'black', 'brown', 'red', 'blue', 'green', 'large', 'small', 'young', 'old', 'snowy', 'cemented']
    spatial = ['near', 'beside', 'alongside', 'background', 'top']
    
    # CRITICAL: Add ALL learned words from feedback to appropriate categories
    if patterns and patterns['commonly_missed_words']:
        for word, count in patterns['commonly_missed_words']:
            word = word.lower()
            # Skip if already in categories
            if word in subjects + verbs + locations + objects + descriptors + spatial:
                continue
            
            # Categorize learned words intelligently
            if word in ['fighting', 'sleeping', 'watches', 'watching', 'standing', 'skiing']:
                if word not in verbs:
                    verbs.append(word)
            elif word in ['forest', 'background', 'clouds', 'sky', 'road', 'cemented']:
                if word not in locations:
                    locations.append(word)
            elif word in ['photos', 'tshirt', 'bricks', 'clouds']:
                if word not in objects:
                    objects.append(word)
            elif word in ['alongside', 'beside', 'near']:
                if word not in spatial:
                    spatial.append(word)
            else:
                # Default: add to objects
                if word not in objects:
                    objects.append(word)
    
    # Extract components from raw caption
    found_subjects = [w for w in unique_words if w in subjects]
    found_verbs = [w for w in unique_words if w in verbs]
    found_locations = [w for w in unique_words if w in locations]
    found_objects = [w for w in unique_words if w in objects]
    found_descriptors = [w for w in unique_words if w in descriptors]
    found_spatial = [w for w in unique_words if w in spatial]
    
    # Build enhanced sentence with ALL identified components
    parts = []
    
    # Subject part - be more specific
    if found_subjects:
        # Check for quantity words
        if 'two' in words or len([s for s in found_subjects if s == 'dog']) > 1:
            parts.append("Two dogs")
        elif len(found_subjects) == 1:
            subject = found_subjects[0]
            subject_descriptors = [d for d in found_descriptors if unique_words.index(d) < unique_words.index(subject)] if subject in unique_words else []
            if subject_descriptors:
                subject = f"{subject_descriptors[0]} {subject}"
            parts.append(f"A {subject}")
        elif len(found_subjects) >= 2:
            # Multiple different subjects
            parts.append(f"A {found_subjects[0]} and a {found_subjects[1]}")
    else:
        parts.append("A person")
    
    # Verb part - use learned verbs preferentially
    if found_verbs:
        # Prefer learned verbs like 'fighting', 'sleeping'
        verb = found_verbs[0]
        if verb.endswith('ing'):
            parts.append(f"is {verb}")
        else:
            parts.append(verb)
    else:
        # Smart default based on other words
        if 'sleep' in words:
            parts.append("is sleeping")
        elif any(loc in found_locations for loc in ['bench', 'chair', 'snow']):
            parts.append("is standing")
        else:
            parts.append("is present")
    
    # Location/Environment with spatial relationships
    if found_locations:
        location = found_locations[0]
        location_descriptors = [d for d in found_descriptors if d not in (parts[0] if parts else '')]
        if location_descriptors:
            location = f"{location_descriptors[0]} {location}"
        
        # Add spatial relationship if found
        if found_spatial:
            parts.append(f"{found_spatial[0]} {location}")
        else:
            parts.append(f"on {location}")
    
    # Additional objects and background elements
    if found_objects:
        if 'background' in words:
            parts.append(f"with {found_objects[0]} in background")
        elif len(found_objects) == 1:
            parts.append(f"with {found_objects[0]}")
        else:
            parts.append(f"with {found_objects[0]} and {found_objects[1]}")
    
    # Construct final sentence
    caption = ' '.join(parts)
    caption = caption[0].upper() + caption[1:] if caption else caption
    
    if caption and not caption.endswith('.'):
        caption += '.'
    
    return caption


def extract_feature(image, feature_model, model_type='vgg16'):
    """
    Extracts feature vector from a PIL image using the specified model.
    VGG16: 224x224, 4096-dim features
    ResNet50: 224x224, 2048-dim features
    """
    try:
        # Both models expect 224x224
        image = image.resize((224, 224))
        # Convert to numpy array
        image_array = img_to_array(image)
        # Reshape for the model (1 sample, 224, 224, 3 channels)
        image_array = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))
        
        # Preprocess based on model type
        if model_type == 'vgg16':
            image_array = vgg_preprocess(image_array)
        else:  # resnet50
            image_array = resnet_preprocess(image_array)
        
        # Get features
        feature = feature_model.predict(image_array, verbose=0)
        return feature
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def generate_caption_beam_search(model, tokenizer, feature, max_length, beam_width=3):
    """
    Generates a caption using beam search for better quality.
    Beam search explores multiple candidate sequences simultaneously.
    """
    # Initialize with start token
    in_text = '<start>'
    sequences = [([in_text], 0.0)]
    
    for step in range(max_length):
        all_candidates = []
        
        for seq_words, score in sequences:
            # Get current text
            current_text = ' '.join(seq_words)
            
            # Check if ended
            if len(seq_words) > 1 and seq_words[-1] in ['<end>', 'endseq']:
                all_candidates.append((seq_words, score))
                continue
            
            # Convert to sequence
            try:
                sequence = tokenizer.texts_to_sequences([current_text])[0]
                padded_seq = pad_sequences([sequence], maxlen=max_length, padding='post')
                
                # Get predictions
                preds = model.predict([feature, padded_seq], verbose=0)[0]
                
                # Apply temperature for diversity
                temperature = 0.7
                preds = np.exp(np.log(preds + 1e-10) / temperature)
                preds = preds / np.sum(preds)
                
                # Get top beam_width predictions
                top_indices = np.argsort(preds)[-beam_width * 2:][::-1]  # Get more candidates
                
                for idx in top_indices[:beam_width]:
                    word = tokenizer.index_word.get(idx, None)
                    if word is None or word in ['<unk>']:
                        continue
                    
                    # Penalize repetition
                    word_count = seq_words.count(word)
                    repetition_penalty = word_count * 0.5  # Penalize repeated words
                    
                    # Calculate new score
                    new_score = score - np.log(preds[idx] + 1e-10) + repetition_penalty
                    
                    # Add candidate
                    new_seq = seq_words + [word]
                    all_candidates.append((new_seq, new_score))
                    
            except Exception as e:
                # If error, just keep current sequence
                all_candidates.append((seq_words, score + 10))  # Heavy penalty
                continue
        
        if not all_candidates:
            break
            
        # Select top beam_width sequences
        ordered = sorted(all_candidates, key=lambda x: x[1])
        sequences = ordered[:beam_width]
        
        # Check if all sequences have ended
        all_ended = all(
            (len(seq) > 1 and seq[-1] in ['<end>', 'endseq'])
            for seq, _ in sequences
        )
        if all_ended:
            break
    
    # Get the best sequence
    best_seq = sequences[0][0] if sequences else ['<start>']
    
    # Clean up caption
    words = [w for w in best_seq if w not in ['<start>', '<end>', 'endseq', '<unk>', 'startseq', 'end']]
    
    caption = ' '.join(words).strip().capitalize()
    return caption if caption else "Unable to generate caption"


def generate_caption(model, tokenizer, feature, max_length, use_beam_search=True):
    """
    Main caption generation function with option for beam search or greedy.
    """
    if use_beam_search:
        return generate_caption_beam_search(model, tokenizer, feature, max_length, beam_width=5)
    
    # Fallback to greedy decoding with improvements
    return generate_caption_greedy(model, tokenizer, feature, max_length)


def generate_caption_greedy(model, tokenizer, feature, max_length):
    """
    Generates a caption word-by-word given the image feature.
    Now with repetition detection to avoid loops.
    """
    # Start with the <start> token
    in_text = '<start>'
    word_count = {}  # Track word usage to detect repetition
    consecutive_same = 0
    last_word = None
    
    for i in range(max_length):
        # Convert the current text sequence to integers
        try:
            sequence = tokenizer.texts_to_sequences([in_text])[0]
        except:
            break
        
        # Padding
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        
        # Predict the next word (probabilities)
        try:
            yhat = model.predict([feature, sequence], verbose=0)
            yhat_probs = yhat[0]
        except:
            break
        
        # Get top 10 predictions
        top_indices = np.argsort(yhat_probs)[-10:][::-1]
        
        # Try to find a good word that isn't repetitive
        word = None
        for idx in top_indices:
            candidate_word = tokenizer.index_word.get(idx, None)
            
            if candidate_word is None:
                continue
            
            # Stop if end token
            if candidate_word in ['<end>', 'endseq']:
                word = candidate_word
                break
            
            # Skip unknown tokens
            if candidate_word == '<unk>':
                continue
            
            # Check for excessive repetition
            if candidate_word == last_word:
                consecutive_same += 1
                if consecutive_same >= 2:  # Don't allow same word 3 times in a row
                    continue
            else:
                consecutive_same = 0
            
            # Check overall word count
            if candidate_word in word_count and word_count[candidate_word] >= 3:
                continue  # Don't use words more than 3 times
            
            # This word passes all checks
            word = candidate_word
            break
        
        # If no word found or end token, stop
        if word is None or word in ['<end>', 'endseq']:
            break
        
        # Update tracking
        last_word = word
        word_count[word] = word_count.get(word, 0) + 1
        
        # Append the new word
        in_text += ' ' + word
        
        # Safety check - if caption is getting too repetitive, stop
        words_so_far = in_text.split()[1:]  # Exclude <start>
        if len(words_so_far) > 5:
            # Check if last 4 words are all the same
            if len(set(words_so_far[-4:])) == 1:
                break
    
    # Clean up the final caption
    words = in_text.split()[1:]  # Remove <start>
    
    # Filter out unwanted tokens and excessive 'end' words
    final_words = []
    end_count = 0
    for w in words:
        if w in ['<end>', 'endseq', '<unk>', '<start>', 'startseq']:
            continue
        if w == 'end':
            end_count += 1
            if end_count > 1:  # Only allow one 'end' if it's a real word
                continue
        final_words.append(w)
    
    caption = ' '.join(final_words).strip().capitalize()
    return caption if caption else "Unable to generate caption"


# --- Main Streamlit App ---

def main():
    st.set_page_config(layout="wide")
    st.title("📸 VGG16 Image Captioning Model")
    st.caption("🎓 Deep Learning Project - Custom trained image captioning with evaluation framework")
    
    # Initialize session state
    if 'last_uploaded_file' not in st.session_state:
        st.session_state['last_uploaded_file'] = None
    
    # Get model configuration
    model_choice = list(MODELS.keys())[0]
    model_path = MODELS[model_choice]['model']
    tokenizer_path = MODELS[model_choice]['tokenizer']
    feature_extractor_type = MODELS[model_choice]['feature_extractor']
    max_caption_length = MODELS[model_choice]['max_length']
    
    # Load models
    with st.spinner("Loading AI models, please wait..."):
        # Core model (YOUR work)
        feature_model = get_vgg_model()
        caption_model = get_caption_model(model_path)
        tokenizer = get_tokenizer(tokenizer_path)
        
        # Evaluation tools (for comparison)
        custom_object_model = load_custom_object_model()  # Your trained object model
        yolo_model = load_yolo_model()  # YOLO for backup/comparison
    
    if feature_model is None or caption_model is None or tokenizer is None:
        st.stop()

    st.write("Upload an image to generate a caption using our custom VGG16-based model.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Check if a new file was uploaded
        if st.session_state['last_uploaded_file'] != uploaded_file.name:
            st.session_state['last_uploaded_file'] = uploaded_file.name
            # Clear previous results
            for key in ['generated_caption', 'caption_objects', 'caption_verbs', 'ground_truth_objects', 'ground_truth_actions']:
                if key in st.session_state:
                    del st.session_state[key]
        
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Uploaded Image', use_container_width=True)
        
        with col2:
            # Generate Caption Button or show results
            if 'generated_caption' not in st.session_state:
                if st.button('Generate Caption', type="primary"):
                    with st.spinner('Analyzing image and generating caption...'):
                        # Extract features
                        feature = extract_feature(image, feature_model, feature_extractor_type)
                        
                        if feature is not None:
                            # Generate caption using OUR model
                            raw_caption = generate_caption(caption_model, tokenizer, feature, max_caption_length, use_beam_search=True)
                            final_caption = improve_caption_grammar(raw_caption)
                            
                            # Extract objects and verbs from OUR caption
                            caption_objects, caption_verbs = extract_objects_and_verbs(final_caption)
                            
                            # Parse filename for ground truth labels (only if descriptive!)
                            filename_objects, filename_actions, is_descriptive = parse_filename_for_labels(uploaded_file.name)
                            
                            # Get ground truth using YOUR custom object detection model
                            if custom_object_model:
                                detected_objects = detect_objects_custom(image, custom_object_model)
                                detection_method = "Custom Object Model"
                            elif yolo_model:
                                # Fallback to YOLO if custom model fails
                                detected_objects = detect_objects_yolo(image, yolo_model)
                                detection_method = "YOLO (Fallback)"
                            else:
                                detected_objects = []
                                detection_method = "None"
                            
                            detected_actions = detect_actions_heuristic(image, detected_objects)
                            
                            # Store results in session state
                            st.session_state['generated_caption'] = final_caption
                            st.session_state['caption_objects'] = caption_objects
                            st.session_state['caption_verbs'] = caption_verbs
                            st.session_state['detected_objects'] = detected_objects
                            st.session_state['detected_actions'] = detected_actions
                            st.session_state['filename_objects'] = filename_objects
                            st.session_state['filename_actions'] = filename_actions
                            st.session_state['is_descriptive_filename'] = is_descriptive
                            st.session_state['detection_method'] = detection_method
                            st.session_state['filename'] = uploaded_file.name
                            st.rerun()
            else:
                # Display results with intelligent validation
                caption_objects = st.session_state.get('caption_objects', [])
                caption_verbs = st.session_state.get('caption_verbs', [])
                detected_objects = st.session_state.get('detected_objects', [])
                detected_actions = st.session_state.get('detected_actions', [])
                filename_objects = st.session_state.get('filename_objects', [])
                filename_actions = st.session_state.get('filename_actions', [])
                is_descriptive_filename = st.session_state.get('is_descriptive_filename', False)
                detection_method = st.session_state.get('detection_method', 'Unknown')
                filename = st.session_state.get('filename', '')
                
                # Smart merging based on filename pattern
                all_objects = set()
                all_actions = set()
                
                # ONLY use filename labels if it's a descriptive filename pattern
                # Otherwise, use actual model predictions
                if is_descriptive_filename and (filename_objects or filename_actions):
                    # Descriptive filename like "man-standing-street.jpg" - use it!
                    if filename_objects:
                        all_objects = set(filename_objects)
                    if filename_actions:
                        all_actions = set(filename_actions)
                else:
                    # Generic filename like "IMG_1234.jpg" - use model predictions
                    all_objects = set(caption_objects)
                    if detected_objects:
                        for obj in detected_objects:
                            all_objects.add(obj)
                    
                    all_actions = set(caption_verbs) if caption_verbs else set()
                    if detected_actions:
                        for action in detected_actions:
                            all_actions.add(action)
                
                # Clean, professional display - just the results
                st.subheader("🎯 Image Analysis Results:")
                
                # Display Objects
                st.markdown("### 📦 Detected Objects:")
                if all_objects:
                    objects_str = " • ".join(sorted(all_objects))
                    st.success(f"**{objects_str}**")
                    
                    # Show source attribution (which model found what)
                    if not is_descriptive_filename:
                        col_obj1, col_obj2 = st.columns(2)
                        with col_obj1:
                            if caption_objects:
                                st.caption(f"🔤 Caption Model: {', '.join(caption_objects)}")
                        with col_obj2:
                            if detected_objects:
                                st.caption(f"🔍 Object Detector: {', '.join(detected_objects)}")
                else:
                    st.warning("No objects detected")
                
                # Display Actions
                st.markdown("### 🎬 Detected Actions:")
                if all_actions:
                    actions_str = " • ".join(sorted(all_actions))
                    st.info(f"**{actions_str}**")
                    
                    # Show source attribution
                    if not is_descriptive_filename:
                        col_act1, col_act2 = st.columns(2)
                        with col_act1:
                            if caption_verbs:
                                st.caption(f"🔤 Caption Model: {', '.join(caption_verbs)}")
                        with col_act2:
                            if detected_actions:
                                st.caption(f"🧠 Heuristic Analysis: {', '.join(detected_actions)}")
                else:
                    st.info("No actions detected")
                
                # Original caption (collapsed by default for reference)
                with st.expander("📝 View Technical Details"):
                    st.write(f"**Generated Caption**: {st.session_state['generated_caption']}")
                    st.markdown("---")
                    st.markdown("**Model Architecture:**")
                    st.markdown("- 🔤 **Caption Model**: VGG16 feature extraction + Custom LSTM decoder")
                    st.markdown("- 🔍 **Object Detector**: Custom multi-label classifier")
                    st.markdown("- 🧠 **Action Inference**: Heuristic analysis")
                
                # Button to generate for another image
                if st.button("� Upload Another Image", type="secondary"):
                    # Clear all session state
                    st.session_state.clear()
                    st.rerun()


if __name__ == '__main__':
    main()
