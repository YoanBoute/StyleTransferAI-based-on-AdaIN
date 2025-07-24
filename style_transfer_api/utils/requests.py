from pathlib import Path
import base64
from pydantic import BaseModel, Field
try :
    from .file import File
except :
    from utils.file import File
import uuid
from enum import Enum

class RequestType(int, Enum) :
    gen = 0
    cancel = 1


class Request(BaseModel) :
    client_id : str
    request_id : str = Field(default_factory = lambda : uuid.uuid4().hex)
    type : RequestType

    @classmethod
    def from_dict(cls, dict) :
        if dict['type'] == RequestType.gen :
            return GenRequest(**dict)
        elif dict['type'] == RequestType.cancel :
            return CancelRequest(**dict)
    
class GenRequest(Request) :
    content_img : File 
    style_imgs : list[File]
    params : dict
    type : RequestType = RequestType.gen

class CancelRequest(Request) :
    type : RequestType = RequestType.cancel


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


    