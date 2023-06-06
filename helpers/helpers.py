from azure.cosmosdb.table.tableservice import TableService
from azure.cosmosdb.table.models import Entity
import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import streamlit as st

class TableManager:
    def __init__(self, account_name, account_key, table_name):
        self.account_name = account_name
        self.account_key = account_key
        self.table_name = table_name

    def write_entry_to_table(self, entry_id, video_name, text, author, length):
        table_service = TableService(account_name=self.account_name, account_key=self.account_key)

        # Create an entity object and set its properties
        entity = Entity()
        entity.PartitionKey = entry_id   # I do not have a good partition key here
        entity.RowKey = "" # row key is not needed with unique partition keys
        entity.Name = video_name
        entity.Text = text
        entity.Author = author
        entity.Length = length

        # Insert the entity into the table
        table_service.insert_or_replace_entity(self.table_name, entity)

    def retrieve_summary_from_table(self, video_id):
        table_service = TableService(account_name=self.account_name, account_key=self.account_key)

        try:
            entity = table_service.get_entity(self.table_name, video_id, "")
            summary = entity.get('Text')
            return summary
        except Exception as e:
            return None
        
    def list_all_entries(self):
        table_service = TableService(account_name=self.account_name, account_key=self.account_key)
        entities = table_service.query_entities(self.table_name)

        # Create an empty list to store the entities
        entity_list = []

        for entity in entities:
            # Create a dictionary with the entity's ID and text
            video_id = entity.PartitionKey
            entity_dict = {'author': entity.get('Author', ''), 'name': entity.get('Name', ''), 'length': entity.get('Length', ''), 'url': self.get_youtube_watch_url(video_id), 'summary': entity.get('Text', '')}

            # Append the dictionary to the list
            entity_list.append(entity_dict)

        # Convert the list of dictionaries to a pandas dataframe
        df = pd.DataFrame(entity_list)

        # Set data types
        df = df.astype({
            'author': 'string',
            'name': 'string',
            'length': 'int',
            'url': 'string',
            'summary': 'string'
        })

        # Return the dataframe
        return df
    
    def get_youtube_watch_url(self, video_id):
        return f"https://www.youtube.com/watch?v={video_id}"
    
# see https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df