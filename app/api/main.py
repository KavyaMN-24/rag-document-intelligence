from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.rag_pipeline import RAGPipeline


UPLOAD_DIRECTORY = Path("data/uploads")
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG pipeline when the API starts."""

    if app.state.pipeline is None:
        app.state.pipeline = RAGPipeline()

    yield


def create_app(pipeline=None) -> FastAPI:
    """Create the FastAPI application."""

    application = FastAPI(
        title="RAG Document Intelligence API",
        description="Ask questions about uploaded PDF documents.",
        version="1.0.0",
        lifespan=lifespan,
    )

    application.state.pipeline = pipeline

    @application.get("/health")
    def health_check():
        return {"status": "ok"}

    @application.post("/documents/upload")
    async def upload_document(
        request: Request,
        file: UploadFile = File(...),
    ):
        """Upload and index a PDF document."""

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Filename is required.",
            )

        safe_filename = Path(
            file.filename.replace("\\", "/")
        ).name

        if Path(safe_filename).suffix.lower() != ".pdf":
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported.",
            )

        # Use a unique directory to prevent filename collisions.
        document_directory = UPLOAD_DIRECTORY / uuid4().hex
        document_directory.mkdir(parents=True, exist_ok=True)

        file_path = document_directory / safe_filename
        total_size = 0

        try:
            with file_path.open("wb") as destination:
                while True:
                    chunk = await file.read(1024 * 1024)

                    if not chunk:
                        break

                    total_size += len(chunk)

                    if total_size > MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail="PDF exceeds the 20 MB size limit.",
                        )

                    destination.write(chunk)

            result = request.app.state.pipeline.index_document(
                file_path
            )

            return {
                "message": "Document indexed successfully.",
                **result,
            }

        except HTTPException:
            file_path.unlink(missing_ok=True)
            raise

        except (ValueError, RuntimeError) as exc:
            file_path.unlink(missing_ok=True)

            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        finally:
            await file.close()

    class QuestionRequest(BaseModel):
        question: str = Field(min_length=1)
        top_k: int = Field(default=5, ge=1, le=20)

    @application.post("/questions/ask")
    def ask_question(
        payload: QuestionRequest,
        request: Request,
    ):
        """Answer a question using indexed documents."""

        try:
            return request.app.state.pipeline.ask(
                question=payload.question,
                top_k=payload.top_k,
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    return application


app = create_app()