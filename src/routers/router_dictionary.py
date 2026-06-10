from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any
import os

from src.db.db_manager import db
from src.utils.logger import logger

router = APIRouter(prefix="/dict", tags=["Dictionary"])

@router.get("/{domain}")
async def get_domain_dictionary(domain: str):
    """Lấy toàn bộ từ điển của một ngành."""
    try:
        data = db.load_dictionary(domain)
        return {"domain": domain, "data": data}
    except Exception as e:
        logger.error(f"Error fetching dictionary for {domain}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{domain}")
async def update_domain_dictionary(domain: str, payload: Dict[str, Any] = Body(...)):
    """
    Cập nhật toàn bộ hoặc thêm mới vào từ điển của một ngành.
    Payload mong đợi: {"Python": {"vi": "Ngôn ngữ lập trình", "pho": "Pai-thơn"}}
    """
    try:
        # Load existing
        data = db.load_dictionary(domain)
        # Update with payload
        data.update(payload)
        # Save
        db.save_dictionary(domain, data)
        return {"status": "success", "message": f"Dictionary for {domain} updated."}
    except Exception as e:
        logger.error(f"Error updating dictionary for {domain}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{domain}/{term}")
async def delete_term_from_dictionary(domain: str, term: str):
    """Xóa một từ khỏi từ điển ngành."""
    try:
        data = db.load_dictionary(domain)
        if term in data:
            del data[term]
            db.save_dictionary(domain, data)
            return {"status": "success", "message": f"Deleted '{term}' from {domain}."}
        else:
            raise HTTPException(status_code=404, detail="Term not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting term from {domain}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
