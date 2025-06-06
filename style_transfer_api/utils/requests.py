from pathlib import Path
import base64
from pydantic import BaseModel, Field
try :
    from .file import File
except :
    from utils.file import File
import uuid
from enum import Enum
    
class Request(BaseModel) :
    client_id : str
    request_id : str = Field(default_factory = lambda : uuid.uuid4().hex)
    content_img : File
    style_imgs : list[File]
    params : dict

    @classmethod
    def from_dict(cls, dict) :
        return cls(**dict)


class GenerationStatus(str, Enum) :
    success = "Success"
    cancel = "Cancelled"
    error = "Error"


class Response(BaseModel) :
    request_id : str
    status : GenerationStatus
    message : str
    generated_image : File | None = None
    
    @classmethod
    def from_dict(cls, dict) :
        return cls(**dict)


    