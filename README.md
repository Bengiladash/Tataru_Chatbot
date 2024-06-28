#Tataru_Chatbot, developed by Benjamin J. Garcia on 6/28/2024, inspired from https://www.youtube.com/watch?v=N_OOfkEWcOk&t=71s

The purpose of this program is to act as an AI assistant that helps users navigate and access the CSUSB knowledge database https://www.csusb.edu/its/support/it-knowledge-base and all of its information. The chatbot is designed to only help with information included
in this database and cannot help with anything else out of that context.

It utilizes Streamlit as the UI/webbrowser service, it utilizes langchain connectors which work in conjunction with nvidia's nim models as the llm, the specific model used is the meta-llama3-70b-instruct model https://build.nvidia.com/meta/llama3-70b.
any amount of custom data can be added on the left sidebar in the app, however try to avoid uploading any given file greater than 20KB as it has proven to throw errors during vector creation when trying to upload any one file that large.
My solution was simply splitting all of my data into files that were less than 20KB to ensure that no payload issues occured.
