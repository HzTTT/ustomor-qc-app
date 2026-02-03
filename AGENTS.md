# Repository Guidelines

## Project Structure & Module Organization
- `app/` contains the FastAPI backend. Key modules: `app/main.py` (routes and views), `app/models.py` (SQLModel models), `app/auth.py` (auth), `app/importers/` (data importers), `app/templates/` (Jinja2 HTML), `app/static/` (CSS/assets), and `app/tools/` (helpers).
- `scripts/` holds operational utilities (e.g., data generation and diagnostics).
- `tests/` contains pytest tests; `docs/` holds implementation notes and runbooks.
- `data/` includes sample import payloads; `docker-compose.yml` defines app/db/worker/cron services.
- `pgdata/` and `qc_*.dump` are local database storage/backup artifacts.

## Build, Test, and Development Commands
- `docker compose up --build`: build and run the full stack (app, db, worker, cron).
- Local dev (from `app/`):
  - `pip install -r requirements.txt`
  - `uvicorn main:app --reload --host 0.0.0.0 --port 8000`
  - `python worker.py` (background worker) and `python cron_runner.py` (scheduled jobs).
- Database migration (from `app/`): `python migrate.py`.
- Diagnostics: `python scripts/test_auto_qc.py` or `docker compose exec app python /app/scripts/test_auto_qc.py`.

## Coding Style & Naming Conventions
- Python: 4-space indentation, `snake_case` functions/vars, `PascalCase` classes, and type hints where practical.
- Keep route handlers centralized in `app/main.py`; factor helpers into nearby modules instead of expanding the route file.
- Templates live in `app/templates/` and static assets in `app/static/`; keep names descriptive (e.g., `reports.html`).
- No formatter is enforced; align with existing style and keep diffs minimal.

## Testing Guidelines
- Framework: pytest. Naming: `tests/test_*.py` and `test_*` functions.
- Example: `pytest tests/test_tag_merge.py -v`.
- Scripted checks in `scripts/` may require a seeded database.

## Commit & Pull Request Guidelines
- Recent history favors conventional commits like `feat(scope): summary` (e.g., `feat(auto-qc): add hourly qc checks`). Keep the subject short and imperative.
- PRs should include: a clear description, test commands run, and screenshots for UI changes. Call out schema/migration or data-import changes explicitly.

## Security & Configuration Tips
- Use `.env.example` as the baseline for required env vars; never commit secrets.
- Rotate `APP_SECRET_KEY` and admin credentials before deployment; see `DATABASE_SETUP.md` for DB setup notes.
