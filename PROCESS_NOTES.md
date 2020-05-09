# Process Notes

## Adding items to requirements.txt

Do:

`pip install package && pip freeze > requirements.txt`

## Adding an environment variable

This looks pretty cool: https://github.com/theskumar/python-dotenv

But instead, I set the SLACK_WEBHOOK environment variable directly inside Render.

## Hitting fastai mismatches again

In the google colab notebook where I built the model for this project, I ran:

```
!pip freeze > requirements.txt
cat requirements.txt
```

Then I took a look at the generated `requirements.txt` file and updated the version numbers for any package there that matched a line in the `requirements.txt` file in this repo:

```
fastai==1.0.61
torch==1.5.0+cu101
torchvision==0.6.0+cu101
numpy==1.18.4
```


