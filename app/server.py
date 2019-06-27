import aiohttp
import asyncio
import uvicorn
import requests
import random
import re
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://qz-aistudio-public.s3.amazonaws.com/rosebot/export-jun19.pkl'  # Update this URL
export_file_name = 'export-jun19.pkl'
classes = ['blowup',
 'dumbwaiter',
 'flag',
 'haring',
 'pipes',
 'pool',
 'pool2',
 'starwall',
 'wallofsound',
 'wallofsound2'] # Update the classes here
 
slack_webhook_url = os.getenv("SLACK_WEBHOOK")
slack_intro_phrases = [
    "I think this is", 
    "I replied that this was", 
    "My response to this photo was", 
    "My computer brain says this is", 
    "According to me, this is", 
    "I'd call this"]
 
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

def predict_this(image_data):
    
    # run the image data against the model
    lets_predict = learn.predict(image_data)
    
    # The best match is the first value in the prediction object
    best_match = str(lets_predict[0])
    
    # The category is the second value in the prediction object
    # which we turn into a number with .item()
    cat_number = lets_predict[1].item()
    
    # A tensor with all of the confidence levels is the third value in the object
    predictions = lets_predict[2]

    # Here I pluck the cat_number'th value from the predictions tensor
    # and make it a number with .item()
    confidence = predictions[cat_number].item()
    
    if confidence > 0.71:
        # lop off the final digit if there is one,
        # for example 'wallofsound' and 'wallofsound2' should return 'wallofsound'
        my_final_answer = re.sub(r'(.*)\d$', r'\1', best_match)
    else:
        my_final_answer = "uncertain"
        
    
    
    response_dict = {
        'best_match': str(best_match),
        'confidence': round(float(confidence),3),
        'result': str(my_final_answer)
        }
    
    return response_dict
        
def slack_this(data, image_url):
    
    if data['result'] == 'uncertain':
        message_color = "#cc0000" # red
    else:
        message_color = "#009933" # green
        
    phrase = random.choice(slack_intro_phrases)
        
    slack_json = {
        'text': f"{image_url}\n{phrase} *{data['result']}*.",
        'attachments': [
            {
                'color': message_color,
                'footer': f"*Confidence*: {data['confidence']}\n*Closest Match*: {data['best_match']}"
            }
        ]
    }
    
    r = requests.post(slack_webhook_url, json=slack_json)
    print(f"Sent to Slack. Response: {r.status_code}") 
    return


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    # get the data from the form submission
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    
    # send the image to my predictor function
    # and return the result whatever asked for it
    data_dict = predict_this(img)
    return JSONResponse(data_dict)
    
    
@app.route('/checkurl', methods=['POST'])
async def checkurl(request):
    # get the json request, ie:
    # {"url":"https://example.com/someimage.jpg"}
    incoming_json = await request.json()
    
    # pull out the url from that request
    url = incoming_json["url"]

    # get the image off the internet
    response = requests.get(url)
    
    # turn it into data
    img = open_image(BytesIO(response.content))
    
    # send the image to my predictor function
    # and return the result whatever asked for it
    data_dict = predict_this(img)
    slack_this(data_dict, url)
    return JSONResponse(data_dict)


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
