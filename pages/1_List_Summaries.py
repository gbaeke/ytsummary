import streamlit as st
import dotenv
import os, sys, pathlib


sys.path.append(str(pathlib.Path().absolute()) + "/helpers")
from helpers import TableManager, filter_dataframe

dotenv.load_dotenv()

# Azure Storage connection information
account_name = os.getenv("STORAGE_ACCOUNT", "")
account_key = os.getenv("STORAGE_KEY", "")
table_name = os.getenv("STORAGE_TABLE", "")

if not account_name or not account_key or not table_name:
    st.write("One or more of the required variables to write summaries to a storage table is empty.")
    st.stop()

#  create instance of TableManager class
tm = TableManager(account_name=account_name, account_key=account_key, table_name=table_name)

# use a wide format for this page (default = centered)
st.set_page_config(page_title="Existing Summaries", page_icon="./favicon.png", layout="wide")

st.title("📼 Existing Summaries")
st.write("Double-click cell to view text or go to the video link.")

# get the data as a dataframe
try:
    df = tm.list_all_entries()
except Exception as e:
    st.write("Could not retrieve previous summaries.")
    st.write(e)
    st.stop()

# check if dataframe is empty
if df.empty:
    st.write("No summaries found.")
    st.stop()

# use Streamlit dataframe and configure each column
# link can be double clicked to open
st.dataframe(
    filter_dataframe(df),
    column_config={
        "author": st.column_config.TextColumn(width="medium"),
        "name": st.column_config.TextColumn(width="medium"),
        "url": st.column_config.LinkColumn(width="medium"),
        "length": st.column_config.NumberColumn(width="small"),
        "summary": st.column_config.TextColumn(width="large" ),
    }
)
