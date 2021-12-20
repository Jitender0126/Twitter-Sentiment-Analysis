
from setuptools import setup

setup(
  name="tweet_sentiment_classifier",
  version="0.1",
  include_package_data=True,
  scripts=["preprocess.py", "model_prediction.py"]
)
