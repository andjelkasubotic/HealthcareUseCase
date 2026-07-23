from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException

from service.dependencies import app_lifespan, get_draft_pipeline, get_draft_store
from service.models import CreateDraftRequest, DraftRecord
from service.pipeline import DraftPipeline
from service.store.base import DraftStore


def create_app() -> FastAPI:
    app = FastAPI(title="Message Handler", version="0.1.0", lifespan=app_lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/drafts", response_model=DraftRecord, status_code=201)
    # store/pipeline are injected via Depends (shared singletons; overridable in tests).
    async def create_draft(
        request: CreateDraftRequest,
        store: DraftStore = Depends(get_draft_store),
        pipeline: DraftPipeline = Depends(get_draft_pipeline),
    ) -> DraftRecord:
        draft = await pipeline.run(request.message)
        return await store.save(draft)

    @app.get("/drafts/{draft_id}", response_model=DraftRecord)
    async def get_draft(
        draft_id: UUID,
        store: DraftStore = Depends(get_draft_store),
    ) -> DraftRecord:
        draft = await store.get(draft_id)
        if draft is None:
            raise HTTPException(status_code=404, detail="Draft not found")
        return draft

    return app


app = create_app()
