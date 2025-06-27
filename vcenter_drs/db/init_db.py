from metrics_db import MetricsDB

if __name__ == "__main__":
    db = MetricsDB(host="localhost", user="drs", password="drs", database="drs")
    db.connect()
    db.init_schema()
    db.close() 