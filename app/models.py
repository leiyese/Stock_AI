from app.extensions import db


class StockData_daily(db.Model):
    __tablename__ = "stockdata_daily"
    id = db.Column(db.Integer, primary_key=True)
    ticker_id = db.Column(db.Integer, db.ForeignKey("tickers.id"), nullable=False)
    date = db.Column(db.DateTime, nullable=False)
    open = db.Column(db.Float, nullable=False)
    high = db.Column(db.Float, nullable=False)
    low = db.Column(db.Float, nullable=False)
    close = db.Column(db.Float, nullable=False)
    volume = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f"StockData_daily: {self.tickers.symbol} {self.date}"


class Ticker(db.Model):
    __tablename__ = "tickers"
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), unique=True, nullable=False)
    stock_data = db.relationship("StockData_daily", backref="tickers", lazy=True)

    def __repr__(self):
        return f"<Ticker {self.symbol}>"
