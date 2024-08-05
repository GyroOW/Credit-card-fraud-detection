from sqlalchemy import create_engine, Column, Integer, String, MetaData, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database connection URI (ensure it is encrypted)
DATABASE_URI = 'postgresql+psycopg2://user:password@host/dbname'

# Create an engine with encrypted connection
engine = create_engine(DATABASE_URI, echo=False)

# Create a base class for declarative models
Base = declarative_base()

class FlaggedTransaction(Base):
    """
    Defines the structure of the flagged_transactions table.
    """
    __tablename__ = 'flagged_transactions'
    id = Column(Integer, primary_key=True)
    transaction_id = Column(String, unique=True)
    flag_reason = Column(String)

# Create all tables
Base.metadata.create_all(engine)

# Session creation
Session = sessionmaker(bind=engine)
session = Session()

def store_flagged_transaction(transaction_id, flag_reason):
    """
    Stores a flagged transaction in the database.
    
    Args:
        transaction_id (str): The ID of the transaction.
        flag_reason (str): The reason why the transaction was flagged.
    """
    flagged_transaction = FlaggedTransaction(transaction_id=transaction_id, flag_reason=flag_reason)
    session.add(flagged_transaction)
    session.commit()
