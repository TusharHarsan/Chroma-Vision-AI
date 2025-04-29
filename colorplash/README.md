# ColorSplash

A Flask web application that colorizes black and white images using DeOldify AI.

![ColorSplash Demo](https://i.ibb.co/2tb9pNr/colorsplash-demo.jpg)

## Features

- Upload black and white or grayscale images
- Automatically colorize images using AI
- View side-by-side comparison of original, grayscale, and colorized versions
- Simple and intuitive web interface

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/ahajin07/colorplash.git
   cd colorplash
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

3. Set up DeOldify (the AI colorization model):
   ```
   git clone https://github.com/jantic/DeOldify.git
   cd DeOldify
   pip install -e .
   cd ..
   ```

4. Download the DeOldify model file:
   - Download the `ColorizeArtistic_gen.pth` file from [this link](https://www.dropbox.com/s/usf7uifrctqw9rl/ColorizeArtistic_gen.pth?dl=0)
   - Place it in the `models` directory

## Directory Structure

```
colorplash/
│
├── app.py                    # Main Flask application
├── colorizer_utils.py        # Utility for colorization
├── models/                   # Model storage
│   └── ColorizeArtistic_gen.pth  # DeOldify model file (needs to be downloaded)
├── static/                   # Static assets
│   └── uploads/              # Image upload storage
├── templates/                # HTML templates
│   └── index.html            # Main app page
├── DeOldify/                 # DeOldify submodule
└── requirements.txt          # Python dependencies
```

## Usage

1. Run the application:
   ```
   python app.py
   ```

2. Open your browser and navigate to `http://127.0.0.1:5000`

3. Upload a black and white image and see the colorized result

## How It Works

ColorSplash uses the DeOldify project, which implements a novel approach to image colorization using self-attention mechanisms in GANs (Generative Adversarial Networks). The app:

1. Accepts user-uploaded images
2. Converts them to grayscale for consistent input
3. Processes them through the DeOldify artistic model
4. Displays the original, grayscale, and colorized versions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project relies on [DeOldify](https://github.com/jantic/DeOldify) by Jason Antic
- Built with Flask 