## TEXT_SUMMARIZATION

A lightweight and extensible **Abstractive Text Summarization** project powered by pretrained transformer models and an interactive UI.

---

### Features

- 📋 Interactive UI for summarizing input text in real-time  
- ⚙️ Easily configurable hyperparameters via `constants.py`  
- 📈 Supports training, evaluation, and inference workflows  
- 🌐 Seamless integration with Weights & Biases and Hugging Face Hub  

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/project-2.git
   cd project-2
   ```
2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Launch the UI

Run the following command to start the web interface:

```bash
python UI/app.py
```

Open your browser and navigate to `http://localhost:8000` to interact with the model.

### 2. Training or Fine-tuning Models

By default, hyperparameters are defined in [`constants.py`](./constants.py). To adjust them and retrain:

1. **Create a new Git branch**

   ```bash
   git checkout -b tune-hyperparams
   ```
2. **Edit hyperparameters**

   * Open `constants.py` and modify values under the `HYPERPARAMETERS` section.
3. **Run training in a Kaggle Notebook**

   * Clone your branch:

     ```bash
     git clone https://github.com/auphong2707/nlp-ner.git
     cd nlp-ner
     git checkout tune-hyperparams
     ```
   * Configure environment variables:

     ```bash
     export WANDB_API_KEY=<your_wandb_key>
     export HUGGINGFACE_API_KEY=<your_huggingface_key>
     ```
   * Execute the appropriate training script:

     ```bash
     python main_<model_name>.py
     ```

---

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
