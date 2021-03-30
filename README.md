Scripts to train a feed-forward fully connected neural network for day-ahead PV forecasting

The model weights are optimized using a back-propogation step

### Install Requirements

```python3 -m pip install -r requirements.txt```

### Help for command line arguments

```python3 fit.py -h```

```python3 predict.py -h```

### Training 

```python3 fit.py --input_file ../solar-dataset.pq --filter_criteria 2017 --num_steps 500 --output_file forecasting_model```

## Prediction

```python3 predict.py --model forecasting_model --input_file solar-dataset.pq --filter_criteria 2018```