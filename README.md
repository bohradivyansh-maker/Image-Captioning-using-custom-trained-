# 📸 VGG16 Image Captioning Model

A Deep Learning-based Image Captioning System with object and action detection, built using TensorFlow/Keras and Streamlit.

## 🎯 Features

- **Custom LSTM Caption Generation**: Trained encoder-decoder model with VGG16 features
- **Object Detection**: Custom multi-label classifier for 80 COCO categories
- **Action Detection**: Heuristic-based action inference from visual context
- **Gender-Neutral Captions**: Bias reduction through post-processing
- **Dual Validation**: Cross-validation between caption model and object detector
- **Interactive Web Interface**: Real-time Streamlit application

## 🏗️ Architecture

### Caption Generation Model
- **Encoder**: VGG16 (pre-trained on ImageNet) - 4096-dim feature extraction
- **Decoder**: Custom LSTM with word embeddings
- **Training**: Transfer learning on Flickr-like dataset
- **Max Length**: 34 tokens
- **Decoding**: Beam search (beam_width=3)

### Object Detection Model
- **Type**: Multi-label CNN classifier
- **Classes**: 80 COCO dataset categories
- **Threshold**: 0.3 confidence
- **Output**: Probability distribution over objects

## 🛠️ Technology Stack

- **Python 3.13**
- **TensorFlow 2.20.0** - Deep learning framework
- **Keras 3.12.0** - High-level neural networks API
- **Streamlit** - Web interface
- **NLTK** - Natural language processing
- **OpenCV** - Image processing
- **YOLOv8** - Backup object detection

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image-captioning-app.git
cd image-captioning-app
```

2. Create virtual environment:
```bash
python -m venv venv
```

3. Activate virtual environment:
- Windows (PowerShell):
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- Linux/Mac:
  ```bash
  source venv/bin/activate
  ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Download NLTK data:
```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4')"
```

## 📂 Project Structure

```
image-captioning-app/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── models/
│   ├── best_model.keras     # Caption generation model
│   └── tokenizer.pkl        # Vocabulary tokenizer
├── README.md
└── .gitignore

../object/
└── object_classifier_final.keras  # Object detection model
```

## 🚀 Usage

1. Navigate to project directory:
```bash
cd image-captioning-app
```

2. Activate virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open browser at `http://localhost:8501`

5. Upload an image and click "Generate Caption"!

## 🎓 How It Works

1. **Image Upload** → User selects an image (JPG/PNG)
2. **Feature Extraction** → VGG16 extracts 4096-dim features
3. **Caption Generation** → LSTM decoder generates word sequence using beam search
4. **Post-Processing** → Gender normalization and grammar improvements
5. **Object Detection** → Custom classifier validates detected objects
6. **Action Inference** → Heuristic rules infer actions from objects
7. **Display Results** → Shows objects, actions, and source attribution

## 📊 Model Training

### Caption Model
- **Dataset**: Flickr8k/30k
- **Loss**: Categorical cross-entropy
- **Optimizer**: Adam
- **Metrics**: BLEU score

### Object Detector
- **Dataset**: COCO (80 classes)
- **Loss**: Binary cross-entropy
- **Architecture**: CNN-based multi-label classifier

## 🎨 UI Features

- Clean, professional interface
- Real-time caption generation
- Source attribution (shows which model detected what)
- Expandable technical details
- Support for descriptive filenames (e.g., "person-riding-bike.jpg")

## 🧠 Key Innovations

- **Beam Search Decoding**: Improved caption quality
- **Gender Normalization**: Reduces bias (man/woman → person)
- **Multi-Model Validation**: Caption + object detector cross-check
- **Smart Filename Parsing**: Extracts labels from descriptive names
- **NLP Integration**: POS tagging and lemmatization

## 📝 Model Files

⚠️ **Note**: Model files (`.keras`, `.pkl`) are not included in the repository due to size constraints.

Required model files:
- `models/best_model.keras` - Caption generation model
- `models/tokenizer.pkl` - Vocabulary tokenizer
- `../object/object_classifier_final.keras` - Object detection model

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Divyansh Bohra - Deep Learning Project

## 🙏 Acknowledgments

- VGG16 architecture from [Simonyan & Zisserman, 2014]
- COCO dataset for object detection training
- Flickr dataset for caption training
- Streamlit for the amazing web framework
