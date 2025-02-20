# Cardiovascular Disease Prediction

## Tools
- NumPy
- Pandas
- Scikit-learn
- Django

## Installation

Create a virtual environment and install the dependencies.

In macOS and Linux:
```zsh
python3 -m venv venv
source venv/bin/activate
```
In Windows:

```zsh
python3 -m venv venv
.\venv\Scripts\activate
```

Install dependecies
```zsh
# pip3 install pandas scikit-learn joblib
# pip3 freeze > requirements.txt
pip3 install -r requirements.txt
```

### How to run Project ( Web App)?
```zsh
python3 manage.py runserver
```

### Input
<!-- ![Input/Output](https://github.com/user-attachments/assets/835417ba-b09c-41ad-882f-557599f2fa28) -->
![Input](docs/Heart-Disease-Prediction.png)

### Output

<center>
  <img src="docs/Prediction.png" width="600" height="400" />
</center>


### How to train model?

```zsh
python3 train/model.py 
```

### How to run trained model?

```zsh
python3 train_script.py
```
