# HAM10000 Skin Cancer Dataset Streamlit Dashboard

This project provides an interactive dashboard for exploring the HAM10000 skin cancer dataset using [Streamlit](https://streamlit.io/). The dashboard allows users to view dataset statistics, class distributions, demographic analysis, data quality, and sample images organized by diagnosis class.

## Features
- Dataset overview and key metrics
- Class distribution and imbalance analysis
- Demographic breakdown (age, gender, body location)
- Data quality checks
- Sample image viewer by diagnosis class

## Directory Structure
```
MedML/
├── app.py                # Main Streamlit dashboard
├── requirements.txt      # Python dependencies
├── dataset/
│   ├── HAM10000_metadata.csv
│   └── images/
│       ├── akiec/
│       ├── bcc/
│       ├── bkl/
│       ├── df/
│       ├── mel/
│       ├── nv/
│       └── vasc/
└── ...
```
- Images are organized in `dataset/images/<class>/` subfolders by diagnosis type.

## Setup Instructions
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/MedML.git
   cd MedML
   ```
2. **Install dependencies**
   Ensure you have Python 3.8+ installed. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the HAM10000 dataset from Kaggle**
   - Go to [HAM10000 Kaggle page](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and download the dataset files (`HAM10000_metadata.csv` and image files).
   - Place `HAM10000_metadata.csv` in the `dataset/` folder.
   - Place all downloaded image files (e.g., ISIC_*.jpg) in `dataset/raw_images/` (create this folder if needed).
4. **Organize images by diagnosis class**
   - Run the provided script to organize images into class folders:
   ```bash
   python organize_images_by_lesion.py
   ```
   - This will create the `dataset/images/<class>/` structure automatically.

## Running the Streamlit App
Launch the dashboard with:
```bash
streamlit run app.py
```
- The app will open in your browser at `http://localhost:8501`.
- Use the sidebar to navigate between analysis sections.

## Notes
- The `dataset/images/` directory is excluded from version control (see `.gitignore`).
- You must download the dataset from Kaggle and organize it locally before running the app.
- If you encounter issues with image loading, verify the directory structure and file permissions.
- For questions or contributions, please open an issue or pull request.

## License
This project is for educational and research purposes. See [LICENSE](LICENSE) for details.
