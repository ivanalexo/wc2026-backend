from fastapi import APIRouter

from app.api.v1.endpoints import health, teams, matches, predictions, groups, stats, simulate, players, admin

router = APIRouter()

router.include_router(health.router)
router.include_router(teams.router)
router.include_router(players.router)
router.include_router(matches.router)
router.include_router(predictions.router)
router.include_router(groups.router)
router.include_router(stats.router)
router.include_router(simulate.router)
router.include_router(admin.router)