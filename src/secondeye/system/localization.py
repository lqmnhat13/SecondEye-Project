"""Vietnamese labels used by the local MVP interface and speech output."""

from __future__ import annotations


VI_LABELS = {
    "person": "người",
    "chair": "ghế",
    "table": "bàn",
    "sofa": "ghế sofa",
    "bed": "giường",
    "backpack": "ba lô",
    "handbag": "túi xách",
    "suitcase": "va li",
    "bottle": "chai",
    "potted_plant": "chậu cây",
    "tv": "ti vi",
    "laptop": "máy tính xách tay",
    "toilet": "bồn cầu",
    "sink": "bồn rửa",
    "refrigerator": "tủ lạnh",
}

VI_DIRECTIONS = {
    "left": "bên trái",
    "center": "phía trước",
    "right": "bên phải",
}

VI_DEPTH_ZONES = {
    "near": "gần",
    "medium": "trung bình",
    "far": "xa",
    "unknown": "không rõ",
}

VI_STATES = {
    "WARMING_UP": "ĐANG KHỞI ĐỘNG",
    "IDLE": "SẴN SÀNG",
    "OBSTACLE": "CẢNH BÁO",
    "READ": "ĐỌC CHỮ",
    "SCENE": "MÔ TẢ CẢNH",
    "QUESTION": "HỎI ĐÁP",
    "ERROR": "LỖI",
}


def localize_label(label: str) -> str:
    return VI_LABELS.get(label, label.replace("_", " "))


def localize_depth_zone(zone: str) -> str:
    return VI_DEPTH_ZONES.get(zone, zone)


def localize_state(state: str) -> str:
    return VI_STATES.get(state, state)
