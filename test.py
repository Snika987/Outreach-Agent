import streamlit as st
st.write("start")

import os, time, json, base64, mimetypes
from email.message import EmailMessage
st.write("stdlib ok")

import requests
from bs4 import BeautifulSoup
st.write("requests + bs4 ok")

from dotenv import load_dotenv
st.write("dotenv imported")
load_dotenv()
st.write("load_dotenv ran")

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
st.write("google auth ok")

from google_auth_oauthlib.flow import InstalledAppFlow
st.write("oauthlib ok")

from googleapiclient.discovery import build
st.write("gmail client ok")

from google import genai
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
st.write("genai client created")

import pandas as pd
st.write("ALL OK")