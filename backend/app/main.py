from fastapi import FastAPI

app = FastAPI(title="Vanguardium API")

@app.get("/")
def root():
    return {"message": "Vanguardium API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}