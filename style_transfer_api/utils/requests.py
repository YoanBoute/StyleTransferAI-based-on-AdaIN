from pathlib import Path
import base64
from pydantic import BaseModel
try :
    from file import File
except :
    from utils.file import File
    
class Request(BaseModel) :
    client_id : str
    content_img : File
    style_imgs : list[File]
    params : dict

class Response(BaseModel) :
    client_id : str
    status : str
    message : str
    generated_image : File | None
    
    