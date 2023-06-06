import streamlit as st
import dotenv
import os
import pytube
import tiktoken
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chat_models import AzureChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from pytube.exceptions import RegexMatchError
import sys, pathlib

# add helpers folder to path (required for Streamlit to find the helpers module)
sys.path.append(str(pathlib.Path().absolute()) + "/helpers")

from helpers import TableManager

# get OpenAI key env
dotenv.load_dotenv()

# encoding for gpt-4
encoding = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# cost per 1000 tokens in France Central in euros
model_costs = {
    'gpt-4': {'input': 0.029, 'output': 0.057},
    'gpt-4-32k': {'input': 0.057, 'output': 0.113},
    'gpt-35-turbo': {'input': 0.001869, 'output': 0.001869}
}


def token_cost(numtokens, priceper1000):
    return numtokens * priceper1000 / 1000

def main():
    # page config
    st.set_page_config(page_title="Summarize YouTube Transcript", page_icon="./favicon.png")

    st.title("📼 Summarize YouTube Transcript")

    # Azure Storage connection information
    account_name = os.getenv("STORAGE_ACCOUNT", "")
    account_key = os.getenv("STORAGE_KEY", "")
    table_name = os.getenv("STORAGE_TABLE", "")

    if not account_name or not account_key or not table_name:
        st.write("One or more of the required variables to write summaries to a storage table is empty.")
        st.stop()

    #  create instance of TableManager class
    tm = TableManager(account_name=account_name, account_key=account_key, table_name=table_name)

    youtube_url = st.text_input("Enter YouTube URL:")
    with st.expander("Options", expanded=False):
        always_use_32k = st.checkbox("Use gpt-4-32k")
        overwrite_existing = st.checkbox("Overwrite existing transcript")
    
    if not st.button("Summarize"):
        st.stop()

    # get thumbnail
    try:
        video = pytube.YouTube(youtube_url)
        video_url = video.thumbnail_url
        video_name = video.title
        video_id = youtube_url.split('=')[-1]
        video_author = video.author
        video_length = video.length
    except RegexMatchError:
        st.error("Invalid YouTube URL")
        st.stop()


    with st.sidebar:
        st.write("📈 Video details")
        st.image(video_url, use_column_width=True)
        summary = tm.retrieve_summary_from_table(video_id)
        if summary:
            st.write("This video has already been summarized")
        if overwrite_existing:
            st.write("Summary will be overwritten")

    # write the summary on the main page and stop
    if summary and overwrite_existing == False:
        st.header("📕 Your Summary")
        st.write(summary)
        st.stop()



    # no summary, or summary and overwrite
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en','en-US', 'en-GB','en-US','de','fr', 'zh'])
    except Exception as e:
        st.error(f"Error loading transcript: {e}")
        st.stop()
    
    transcript = ' '.join([t['text'] for t in transcript])

    # check the transcript
    with st.sidebar:
        # calculate tokens in input
        input_tokens = num_tokens_from_string(transcript, 'cl100k_base')

        # depending on the number of tokens, switch deployment
        if input_tokens < 4096:
            model = "gpt-35-turbo"  # this model needs to be deployed for your endpoint
        elif input_tokens > 4096 and input_tokens < 8192:
            model = "gpt-4"
        else:
            model = os.getenv("DEPLOYMENT", "gpt-4") # use the model from the environment and default to gpt-4

        if always_use_32k:
            model = "gpt-4-32k"

        # inform the user about the model
        st.write(f"Using model: {model}")

        # set the input and output costs from the costs in the model_costs dict
        input_cost = model_costs[model]['input']
        output_cost = model_costs[model]['output']

        # add transcript info to the sidebar
        with st.expander("YouTube transcript", expanded=False):
            st.write(f"Number of tokens in transcript: {input_tokens}")
            st.write(f"Cost of transcript tokens: €{token_cost(input_tokens, input_cost):.3f}")
            st.write(transcript)

    # use Azure chat model; requires gtp-4 deployment on given endpoint
    model = AzureChatOpenAI(
        client=None,
        openai_api_base=os.getenv("ENDPOINT", ""),
        openai_api_key=os.getenv("API_KEY", ""),
        openai_api_version="2023-03-15-preview",
        deployment_name=model,
        openai_api_type="azure",
        max_tokens=1024,
        temperature=0,
        verbose=True
    )

    system_template = "You are a helpful assistant that summarizes Youtube transcripts"
    system_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = "Summarize the transcript of this video. Always use English even if the transcript is in another language. If there are different sections, provide more details about each section: {transcript}"
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    with st.spinner("Getting summary..."):
        try:
            chain = LLMChain(llm=model, prompt=chat_prompt)
            result = chain.run(transcript=transcript)
        except Exception as e:
            st.error(f"Error creating summary: {e}")
            st.stop()

    # cost of the result
    output_tokens = num_tokens_from_string(result, 'cl100k_base')

    with st.expander("Tokens and cost", expanded=True):
        st.write(f"Number of transcript tokens: {input_tokens}")
        st.write(f"Number of summary tokens: {output_tokens}")
        st.write(f"Cost of transcript tokens: €{token_cost(input_tokens, input_cost):.3f}")
        st.write(f"Cost of the summary: €{token_cost(output_tokens, output_cost):.3f}")
        st.write(f"Total cost of transcript and summary: €{token_cost(input_tokens, input_cost) + token_cost(output_tokens, output_cost):.3f}")

    st.header("📕 Your Summary")
    st.write(result)

    st.header("👀 Watch the Video")
    st.video(youtube_url)

    with st.sidebar:
        # save to Azure table
        try:
            tm.write_entry_to_table(video_id, video_name, result, video_author, video_length)
            st.success("Summary saved to Azure table")
        except Exception as e:
            st.error(f"Error saving summary: {e}")

if __name__ == "__main__":
    main()