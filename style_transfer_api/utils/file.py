from pathlib import Path
import base64
from pydantic import BaseModel
from datetime import datetime
import re
import mimetypes

class File(BaseModel) :
    filename : str
    extension : str
    data : str

    @classmethod
    def from_path(cls, file : Path) :
            return cls(
                filename = file.stem,
                extension = file.suffix,
                data = base64.b64encode(file.read_bytes()).decode("utf-8")
            )
    
    @classmethod
    def from_dict(cls, file : dict) :
        return cls(**file)
    
    @classmethod
    def from_url(cls, file : str) :
        return cls(
            filename = f'tmp_{datetime.now()}',
            extension = mimetypes.guess_extension(re.findall(r'data:(.*);', file)[0]),
            data = re.findall(r'base64,(.*)', file)[0]
        )
    
    def save_to(self, dst_path : Path) :
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        print("yes")
        dst_path = dst_path.with_suffix(self.extension)
        with open(dst_path, 'wb') as f :
            print("no")
            f.write(base64.b64decode(self.data))
        print("allright")
        return dst_path

