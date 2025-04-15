from pathlib import Path
import base64
from pydantic import BaseModel

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
    
    def save_to(self, dst_path : Path) :
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path = dst_path.with_suffix(self.extension)
        with open(dst_path, 'wb') as f :
            f.write(base64.b64decode(self.data))
        return dst_path

