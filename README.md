## Dependencies
Dependencies used for this code are available via `pip`, EXCEPT the atc_toolbox module.

This can be installed from this repository:  https://github.com/anerli/atc-toolbox.

## Example Usage

Example command to create a new model trained on MSFT:
```
python train.py -m msft_model -e 1000 -s MSFT -n
```

Then can go train the same model on something else too:
```
python train.py -m msft_model -e 1000 -s AMD
```

Training and testing the same model on all data:
```
python train.py -m full_model -a -n -e 2000
python train.py -m full_model -a -t 
```