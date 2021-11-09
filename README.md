# ANLP A2

## Initial Setup
### setup
`virtualenv` HIGHLY recommended.
```
virtualenv -p python3.8 venv
source venv/bin/activate

pip install -r requirements.txt
```

### Downloading files
From the repo root, run 
```
./prepare-inference.sh
```
That should download 1 file.

### Setting PYTHONPATH
Then, run
```
source .env
```
This should set your PYTHONPATH

## How To
### Get embeddings of a word in different contexts
Run
```
python anlp_a2.inference
```

In the first prompt, enter number of sentences (needs to be >= 2).
Then, enter the sentence and the focussed word.

Once you do that, euclidian and cosine distances will be displayed
