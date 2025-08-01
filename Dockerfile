FROM codercom/code-server:latest
EXPOSE 8080
CMD ["code-server", "--host", "0.0.0.0", "--port", "8080"]
