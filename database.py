from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URI = 'postgresql+psycopg2://user:password@host/dbname'

engine = create_engine(DATABASE_URI, echo=False)
Base = declarative_base()

class FlaggedTransaction(Base):
    __tablename__ = 'flagged_transactions'
    id = Column(Integer, primary_key=True)
    transaction_id = Column(String, unique=True)
    flag_reason = Column(String)

Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

def store_flagged_transaction(transaction_id, flag_reason):
    flagged_transaction = FlaggedTransaction(transaction_id=transaction_id, flag_reason=flag_reason)
    session.add(flagged_transaction)
    session.commit()
