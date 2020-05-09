# Process Notes

## Adding items to requirements.txt

Do:

`pip install package && pip freeze > requirements.txt`

## Adding an environment variable

This looks pretty cool: https://github.com/theskumar/python-dotenv

But instead, I set the SLACK_WEBHOOK environment variable directly inside Render.

## Hitting fastai mismatches again

The version of fastai (and torch, torchvision & numpy) I used when training the model need to match the versions used when running the model. 

After more messing with the requirements.txt file, I finally just added `pip install fastai` to the Docker file.

Just noting here for reference that the log files indicate the following torches are installed

```
torch-1.5.0-cp37-cp37m-manylinux1_x86_64.whl
torchvision-0.6.0-cp37-cp37m-manylinux1_x86_64.whl
```

Also noting that it says it's intstalling `... torch-1.5.0 torchvision-0.6.0 ...`




