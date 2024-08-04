from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData
import pandas as pd

def store_flagged_transactions(flagged_transactions):
    """
    Store flagged transactions in a database.

    Args:
    - flagged_transactions (pandas.DataFrame): DataFrame containing flagged transactions.
    """
    engine = create_engine('sqlite:///fraud_transactions.db', echo=False)
    metadata = MetaData()

    # Define a table to store flagged transactions
    flagged_table = Table('flagged_transactions', metadata,
                          Column('index', Integer, primary_key=True),
                          Column('transaction_id', String),
                          Column('flag_reason', String))

    metadata.create_all(engine)

    # Connect to the database and store flagged transactions
    with engine.connect() as conn:
        flagged_transactions.to_sql('flagged_transactions', con=conn, if_exists='append', index_label='index')
