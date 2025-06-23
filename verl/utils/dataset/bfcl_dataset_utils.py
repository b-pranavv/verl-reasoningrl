import os
from typing import List, Dict
from huanzhi_utils import load_file
import json


current_directory = os.path.dirname(os.path.abspath(__file__))

INVOLVED_CLASS_TO_FUNC_DOC_PATH = {
    "GorillaFileSystem": f"{current_directory}/bfcl_tools/gorilla_file_system.json",
    "MathAPI": f"{current_directory}/bfcl_tools/math_api.json",
    "MessageAPI": f"{current_directory}/bfcl_tools/message_api.json",
    "TwitterAPI": f"{current_directory}/bfcl_tools/posting_api.json",
    "TicketAPI": f"{current_directory}/bfcl_tools/ticket_api.json",
    "TradingBot": f"{current_directory}/bfcl_tools/trading_bot.json",
    "TravelAPI": f"{current_directory}/bfcl_tools/travel_booking.json",
    "VehicleControlAPI": f"{current_directory}/bfcl_tools/vehicle_control.json",
}

def construct_tools_from_involved_classes(involved_classes: List[str]) -> str:
    tools = []
    for class_name in involved_classes:
        func_doc = load_file(INVOLVED_CLASS_TO_FUNC_DOC_PATH[class_name])
        for func in func_doc:
            func["description"] = func["description"].split("Tool description: ")[1]
        func_doc = [json.dumps(func) for func in func_doc]
        tools.extend(func_doc)
    return "\n".join(tools)
