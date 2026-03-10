# Tourism Experience Analytics

A beginner-friendly Machine Learning project using Python and Streamlit.

## What it does
- **Classifies** visit mode (Business / Couples / Family / Friends / Solo)
- **Predicts** ratings (1-5 stars) visitors will give attractions
- **Recommends** attractions using Collaborative Filtering, Content-Based Filtering, and Hybrid

---

## Files in this project

| File | Purpose |
|------|---------|
| `train_tourism_model.py` | Trains all ML models - run this first |
| `app.py` | Streamlit web app - run this second |
| `requirements.txt` | Python libraries needed |
| `.gitignore` | Files to exclude from GitHub |

### Data files needed (not pushed to GitHub)
| File | Description |
|------|-------------|
| `Transaction.xlsx` | All visits (52,930 rows) |
| `User.xlsx` | User info |
| `Updated_Item.xlsx` | Attractions (use this, not Item.xlsx) |
| `City.xlsx`, `Country.xlsx`, `Region.xlsx`, `Continent.xlsx` | Geography |
| `Type.xlsx`, `Mode.xlsx` | Lookup tables |

---

## How to run

### 1. Install libraries
```bash
pip install -r requirements.txt
```

### 2. Put all Excel files in the same folder

### 3. Train the models (run once)
```bash
python train_tourism_model.py
```
This creates `.pkl` model files and charts. Takes 3-5 minutes.

### 4. Launch the app
```bash
streamlit run app.py
```

---

## How to push to GitHub

```bash
git init
git add train_tourism_model.py app.py requirements.txt README.md .gitignore
git commit -m "first commit - tourism analytics project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

**Note:** `.pkl` files and `.xlsx` data files are in `.gitignore` because they are large.
If you want to push data files too, remove them from `.gitignore`.

---

## Model Accuracy

| Task | Model | Accuracy |
|------|-------|----------|
| Visit Mode (Classification) | RandomForest / GradientBoosting | ~98% |
| Rating (Regression) | RandomForest / GradientBoosting | R² ~0.998 |

---

