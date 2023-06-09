# Generate YouTube Summaries with GPT

This is a Python Streamlit app that asks for a URL to a YouTube video. Both standard and watch URLs are supported (e.g., https://www.youtube.com/watch?v=VIDEO_ID or https://youtu.be/VIDEO_ID).

The app will retrieve the YouTube transcript for the video and summarize it with either gpt-35-turbo, gpt-4, or gpt-4-32k. If the video is too long and the transcript passes 32k tokens, you will receive an error. There are ways to work around this limitation but they are not implemented in this app.

Every summary is saved to an Azure Storage Table for reference. If you ask for a summary a second time, you get the summary from table storage unless you override that from the options.

➡️ A full description of the app can be found here: https://gpt-inity.addpotion.com/youtube-summarizer

A look at the UI:

https://www.youtube.com/watch?v=sSjalVRxuh8

## Requirements

This app uses Azure OpenAI service and requires that you do three deployments in your selected region (I am using France Central):

- gpt-35-turbo
- gpt-4
- gpt-4-32k

The deployments should be called exactly as above.

⚠️ You need to request access to Azure OpenAI service first. After you have been given access, you have to request access to gpt-4 which gives you both the standard gpt-4 and gpt-4 with 32k context.

The app can easily be changed to support OpenAI directly, but you need to be granted API access to gpt-4 as well.

## Price

Note that gpt-4 and especially gpt-4-32k can be costly. It is not uncommon to summarize a longer YouTube at a cost of about 2 euros. The app gives an indication about the costs based on pricing information in a Python dictionary. The prices in the code were added in June 2023.

## Using the app

Create an Azure OpenAI service with the three deployment listed above. Get the API key and endpoint, you will need that later.

Create a storage account and create a table called `summaries`. Get the storage account name and key. You will need it later.

Steps:

- Clone the repo
- From the cloned folder, run `pip install -r requirements.txt` (requires Python and pip, app was tested with Python 3.11.2)
    - or use a virtual environment (recommended)
- Create a `.env` file in the folder with the following entries:
    - API_KEY=AzureOpenAIAPIKey
    - ENDPOINT=https://NAMEOFYOURSERVICE.openai.azure.com/
    - DEPLOYMENT="gpt-4-32k"
    - STORAGE_ACCOUNT=NAMEOFYOURSTORAGEACCOUNT
    - STORAGE_KEY=STORAGEKEY
    - STORAGE_TABLE=summaries
- Run `streamlit run Create_Summary.py` to open the app in the browser

Paste in a YouTube URL to verify. Use the `List Summaries` page (link in the sidebar) to see that summaries are saved to the table.

## Using Docker

If you have Docker installed, run the following command to create an image for the app:

 `docker build -t <DOCKERID>/image:tag` .

E.g., `docker build -t gbaeke/ytsum:v1`

You can then push the image to Docker Hub with `docker push <DOCKERID>/image:tag` and use the app in Azure Container Apps, ACI etc...

To run the app, use the following command:

```
docker run -p 8501:8501 -e API_KEY=YOURKEY -e ENDPOINT=YOURENDPOINT \
    -e DEPLOYMENT=gpt-4-32k -e STORAGE_ACCOUNT=YOURACCOUNT \
    -e STORAGE_KEY=YOURKEY -e STORAGE_TABLE=YOURTABLE gbaeke/yt
```

Now open `http://localhost:8501` to view the app.