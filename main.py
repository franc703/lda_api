from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import logging

# Import LDA functions
from lda_helper.lda_model import preprocess, run_lda

app = FastAPI()


class TextInput(BaseModel):
    text: str = Field(
        ..., example="This is a sample text.", description="The text to be analyzed."
    )


@app.post(
    "/lda",
    response_description="The topics identified by LDA.",
    responses={
        200: {"description": "Successful Response"},
        500: {"description": "An error occurred during processing."},
    },
)
def analyze_text(input_data: TextInput):
    """
    Analyze the input text and return the topics identified by LDA.
    """
    text = input_data.text

    try:
        # Preprocess
        doc, _ = preprocess(text)

        # Run LDA
        theta = run_lda(doc)
    except Exception as e:
        logging.error("An error occurred: %s", e)
        raise HTTPException(
            status_code=500, detail="An error occurred during processing."
        ) from e

    return {"topics": theta.tolist()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
