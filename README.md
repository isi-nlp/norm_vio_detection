# Norm Violation Detection
This is the repo for norm violation detection on Reddit.

## Data
Decompress the data file
```angular2html
tar -xzf yourfile.tar.gz
```


## Checkpoint 
The BERTRNN model checkpoint is accessible from this Google Drive [link](https://drive.google.com/file/d/1IeRCFlrZw2JKYO0a8M_R3ibsTRd2EE3F/view?usp=sharing).
Download the model to *ckps/BERTRNN*.

The GPT model checkpoint is accessible from this Google Drive link. (to be released soon!)
Download the model to *ckps/GPT*.


## Using the API
Instantiate the API:
```angular2html
python api-inference.py --model=BERTRNN
```
If GPUs are available, specify the GPU(s) to use:
```angular2html
python api-inference.py --model=BERTRNN --gpu=0
```

Then prepare the query data as in *api-test-data.json*.

Calling the API:
```angular2html
curl -X POST -H 'Content-Type: application/json' -d '@api-test-data.json' http://localhost:5000/api
```

