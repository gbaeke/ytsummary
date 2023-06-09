import streamlit as st
import dotenv
import os
import pytube
import tiktoken
from youtube_transcript_api import YouTubeTranscriptApi
from pytube.exceptions import RegexMatchError
import sys, pathlib
import re

# add helpers folder to path (required for Streamlit to find the helpers module)
sys.path.append(str(pathlib.Path().absolute()) + "/helpers")

import helpers

# get OpenAI key env
dotenv.load_dotenv()

def main():
    # page config
    st.set_page_config(page_title="Summarize YouTube Video", page_icon="./favicon.png")

    st.title("📼 Summarize YouTube Video")

    # Azure Storage connection information
    account_name = os.getenv("STORAGE_ACCOUNT", "")
    account_key = os.getenv("STORAGE_KEY", "")
    table_name = os.getenv("STORAGE_TABLE", "")

    if not account_name or not account_key or not table_name:
        st.write("One or more of the required variables to write summaries to a storage table is empty.")
        st.stop()

    # OpenAI infiorma
    endpoint = os.getenv("ENDPOINT", "")
    apikey = os.getenv("API_KEY", "")
    deployment = os.getenv("DEPLOYMENT", "")

    if not endpoint or not apikey or not deployment:
        st.write("One or more of the required variables to connect to OpenAI is empty.")
        st.stop()
    

    #  create instance of TableManager class
    tm = helpers.TableManager(account_name=account_name, account_key=account_key, table_name=table_name)

    youtube_url = st.text_input("Enter YouTube URL:")
    with st.expander("Options", expanded=False):
        use_default = st.checkbox("Use default")
        overwrite_existing = st.checkbox("Overwrite existing transcript")
    
    if not st.button("Summarize"):
        st.stop()

    # get thumbnail
    try:
        video = pytube.YouTube(youtube_url)
        video_url = video.thumbnail_url
        video_name = video.title
        video_id = helpers.get_video_id(youtube_url)
        video_author = video.author
        video_length = video.length
    except RegexMatchError:
        st.error("Invalid YouTube URL")
        st.stop()


    with st.sidebar:
        st.write("📈 Video details")
        st.write(f"{video_name}")
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
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en','nl','en-US', 'en-GB','en-US','de','fr', 'zh'])
    except Exception as e:
        st.error(f"Error loading transcript: {e}")
        st.stop()
    
    transcript = ' '.join([t['text'] for t in transcript])

    # check the transcript
    with st.sidebar:
        # calculate tokens in input
        input_tokens = helpers.num_tokens_from_string(transcript, 'cl100k_base')
        max_tokens = 1024

        # get the model, could be null if input is too long
        model = helpers.get_model(input_tokens, max_tokens)

        if model is None:
            st.error("Input too long, please try a shorter video")
            st.stop()

        if use_default:
            model = deployment  # comes from env var DEPLOYMENT

        # inform the user about the model
        st.write(f"Using model: {model}")

        # set the input and output costs from the costs in the model_costs dict
        input_cost = helpers.model_costs[model]['input']
        output_cost = helpers.model_costs[model]['output']

        # add transcript info to the sidebar
        with st.expander("YouTube transcript", expanded=False):
            st.write(f"Number of tokens in transcript: {input_tokens}")
            st.write(f"Cost of transcript tokens: €{helpers.token_cost(input_tokens, input_cost):.3f}")
            st.write(transcript)

    

    with st.spinner("Getting summary..."):
        try:
            summary = helpers.get_summary(transcript, max_tokens, model, endpoint,
                                          apikey, "azure", False )
        except Exception as e:
            st.error(f"Error creating summary: {e}")
            st.stop()

    # cost of the result
    output_tokens = helpers.num_tokens_from_string(summary, 'cl100k_base')

    with st.expander("Tokens and cost", expanded=True):
        st.write(f"Number of transcript tokens: {input_tokens}")
        st.write(f"Number of summary tokens: {output_tokens}")
        st.write(f"Cost of transcript tokens: €{helpers.token_cost(input_tokens, input_cost):.3f}")
        st.write(f"Cost of the summary: €{helpers.token_cost(output_tokens, output_cost):.3f}")
        st.write(f"Total cost of transcript and summary: €{helpers.token_cost(input_tokens, input_cost) + helpers.token_cost(output_tokens, output_cost):.3f}")

    st.header("📕 Your Summary")
    st.write(summary)

    st.header("👀 Watch the Video")
    st.video(youtube_url)

    with st.sidebar:
        # save to Azure table
        try:
            tm.write_entry_to_table(video_id, video_name, summary, video_author, video_length)
            st.success("Summary saved to Azure table")
        except Exception as e:
            st.error(f"Error saving summary: {e}")

if __name__ == "__main__":
    main()